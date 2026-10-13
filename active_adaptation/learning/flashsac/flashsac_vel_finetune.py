"""Alternate perception training and SAC with frozen perception.

Only the short Stage 1 rollout contains depth/hidden states. Stage 2 replay stores the
teacher replay fields followed by velocity commands and the frozen student latent.
Actions keep the teacher's residual/absolute convention, including ref_joint_pos_.
"""

import copy
from dataclasses import dataclass
from numbers import Integral
from typing import List

import torch
from torch import nn, optim
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import Unbounded
from torchrl.envs.transforms import TensorDictPrimer

from ..ppo.common import ACTION_KEY, OBS_KEY, CatTensors, make_batch, make_conv
from .flashsac_vel import FlashSACVel, FlashSACVelConfig
from .flashsac_vel_flat import ACTION_DR_KEY, U_KEY, FlashSACVelFlat, NStepTrajectoryReplay
from .flashsac_upstream.agent import _sample_flashsac_actions, _update_networks
from .flashsac_upstream.scheduler import ramp_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .finetune_schedule import AdaptivePerceptionSchedule, validate_config as validate_adaptive_config
from ..ppo.ppo_vel import (
    DEPTH_KEY, VEL_CMD_KEY, OBJECT_KEY, OBJECT_PRED_KEY, OBJECT_PRED_TRANS_KEY,
    PRIV_FEATURE_KEY, PRIV_PRED_KEY, REF_JPOS_KEY, TemporalDepthGRU,
)
from ..modules.rnn import set_recurrent_mode


_ADAPTIVE_SCHEDULE_FIELDS = (
    "perception_trigger", "perception_error_window", "perception_trigger_patience",
    "plateau_window", "plateau_tol", "plateau_patience", "plateau_check_interval",
    "cycle_perception_iters_min", "cycle_perception_iters_max",
    "cycle_sac_iters_min", "cycle_sac_iters_max",
)


@dataclass
class FlashSACVelFinetuneConfig(FlashSACVelConfig):
    _target_: str = "active_adaptation.learning.flashsac.flashsac_vel_finetune.FlashSACVelFinetune"
    name: str = "flashsac_vel_finetune"
    num_envs: int = 4096
    # Depth comes from CUDA ray casting; headless training needs no RTX renderer.
    # train.py enables rendering again for eval_render=true.
    enable_rtx: bool = False
    updates_per_interaction_step: float = 8.0
    # Adaptive mode: teacher perception -> SAC, then adaptive SAC/refresh cycles.
    # The first error reference comes from student replay filling, with no extra phase.
    # False preserves the fixed num_cycles schedule below.
    cycle_mode: bool = True
    # Pilot candidate from frozen-actor probes, not a guaranteed performance margin.
    perception_trigger: float = 0.03
    # Aggregate rollout MSE over this many iterations before taking its square root.
    perception_error_window: int = 20
    perception_trigger_patience: int = 3
    # Compare two adjacent windows every check_interval; stop after patience checks
    # with less than plateau_tol relative RMSE improvement (including deterioration).
    plateau_window: int = 40
    plateau_tol: float = 0.0018
    plateau_patience: int = 3
    plateau_check_interval: int = 10
    cycle_perception_iters_min: int = 1
    cycle_perception_iters_max: int = 300
    cycle_sac_iters_min: int = 100
    cycle_sac_iters_max: int = 1000
    # At the nominal training budget, finish any current refresh naturally, then
    # require this many uninterrupted SAC phase iterations (including replay fill).
    # An existing SAC phase counts toward this length; 0 disables the final tail.
    final_sac_iters: int = 200
    # Initial teacher-rollout perception training, in both modes.
    perception_iters: int = 500
    # Fixed mode only: first SAC, followed by num_cycles repeats and final SAC.
    # Adaptive mode ignores all four fixed schedule settings.
    sac_iters: int = 1000
    # Repeated cycles after the first: a perception phase of cycle_perception_iters, then a SAC phase of
    # cycle_sac_iters. num_cycles counts these repeats only; 0 = one-shot perception phase.
    # After the last repeat, SAC runs to the end of training.
    num_cycles: int = 10
    cycle_perception_iters: int = 300
    cycle_sac_iters: int = 500
    # Repeated perception phases use the frozen student SAC actor and train perception only.
    # Stage 1 rollout noise: u = tanh(mean + std * noise * scale). Stage 2 SAC always uses 1.0.
    teacher_perception_rollout_noise_scale: float = 1.6  # initial perception phase
    student_perception_rollout_noise_scale: float = 1.0  # repeated perception phases
    adapt_epochs: int = 2
    perception_ema_tau: float = 0.08
    in_keys: List[str] = (*FlashSACVelConfig.in_keys, DEPTH_KEY)
    # Fixed mode: decay over the estimated SAC update budget. Adaptive mode: use
    # global environment-step progress after teacher perception, including refreshes.
    learning_rate_end: float = 1e-5
    buffer_max_length: int = 10_000_000
    # Rows before SAC updates start, after every replay reset (first SAC phase and each cycle).
    # ~one episode per env (4096 envs x ~635 steps), so updates never see only reset states.
    buffer_min_length: int = 2_600_000
    buffer_device_type: str = "cpu"


ConfigStore.instance().store(
    "flashsac_vel_finetune", node=FlashSACVelFinetuneConfig, group="algo"
)


class FinetuneRollout(TensorDictModuleBase):
    def __init__(self, policy, mode):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.train_mode = mode == "train"
        self.in_keys = list(dict.fromkeys([
            *policy.replay_env_keys, VEL_CMD_KEY, DEPTH_KEY,
            "is_init", "adapt_hx", "depth_hx",
        ]))
        self.out_keys = [ACTION_KEY, U_KEY, PRIV_PRED_KEY,
                         ("next", "adapt_hx"), ("next", "depth_hx")]

    @torch.no_grad()
    def forward(self, td):
        p = self.policy
        if not p._checkpoint_loaded:
            raise RuntimeError("flashsac_vel_finetune requires checkpoint_path to a teacher or finetune checkpoint")
        if ACTION_DR_KEY in p.replay_keys:
            td[ACTION_DR_KEY] = p.action_dr()
        p._perceive(td, ema=True)
        teacher_rollout = self.train_mode and p.stage == 1 and not p.is_repeated_perception_phase
        if teacher_rollout:
            network, observation = p._actor, p.replay_obs(td)
        else:
            network = p._student if p._student is not None else p._student_rollout
            observation = p._actor_adapt_input(td)
        if self.train_mode:
            # Even with an empty replay, sample the warm-start policy (never uniform actions).
            p._cached_noise, u, p._cur_noise_repeat_count, p._cur_noise_repeat_n = (
                _sample_flashsac_actions(
                    actor=network, noise=p._cached_noise, observations=observation,
                    temperature=p._rollout_noise_scale(teacher_rollout),
                    cur_count=p._cur_noise_repeat_count,
                    cur_n=p._cur_noise_repeat_n, zeta_cdf=p._zeta_cdf,
                )
            )
        else:
            mean, _ = network.apply("get_mean_and_std", observation, training=False)
            u = torch.tanh(mean)
        td[U_KEY] = u
        td[ACTION_KEY] = p._joint_action(u, td)
        return td


class FlashSACVelFinetune(FlashSACVel):
    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.perception_iters < 0 or cfg.adapt_epochs < 1:
            raise ValueError("perception_iters must be >= 0 and adapt_epochs >= 1")
        if cfg.cycle_mode:
            validate_adaptive_config(cfg)
            if (isinstance(cfg.final_sac_iters, bool)
                    or not isinstance(cfg.final_sac_iters, Integral) or cfg.final_sac_iters < 0):
                raise ValueError("final_sac_iters must be a nonnegative integer")
        else:
            schedule = (cfg.sac_iters, cfg.cycle_perception_iters, cfg.cycle_sac_iters)
            if cfg.num_cycles < 0 or min(schedule) < 0:
                raise ValueError("num_cycles, sac_iters and cycle_*_iters must be >= 0")
            if cfg.num_cycles > 0 and min(schedule) < 1:
                raise ValueError("repeated cycles need sac_iters, cycle_perception_iters and "
                                 "cycle_sac_iters >= 1")
        for name in ("teacher_perception_rollout_noise_scale", "student_perception_rollout_noise_scale"):
            if not 0 <= getattr(cfg, name) < float("inf"):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 < cfg.perception_ema_tau <= 1:
            raise ValueError("perception_ema_tau must be in (0, 1]")
        if cfg.adapt_module != "gru" or not cfg.use_object_adapt:
            raise ValueError("flashsac_vel_finetune requires GRU and object adaptation")
        if not cfg.enable_residual_distillation:
            raise ValueError("Stage 1 requires enable_residual_distillation=true")
        if observation_spec.get(DEPTH_KEY, None) is None:
            raise ValueError("Student finetune requires depth observations and enabled cameras")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if self.num_envs < cfg.adapt_num_minibatches or self.num_envs % cfg.adapt_num_minibatches:
            raise ValueError("num_envs must be divisible by adapt_num_minibatches")
        self.stage = 1
        self.perception_iters_completed = 0
        self.rl_phase_iters_completed = 0
        self.cycles_completed = 0
        # Explicit phase flag also supports older checkpoints with an initial refresh.
        self._student_perception_phase = False
        self._adaptive_schedule = AdaptivePerceptionSchedule(cfg) if cfg.cycle_mode else None
        self._initial_reference_pending = bool(cfg.cycle_mode)
        self._perception_error_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._perception_error_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self._perception_iteration_ready = False
        self._last_schedule_info = {}
        self._last_transition_reason = "initial_teacher"
        self._adaptive_lr_start_iter = cfg.perception_iters
        self._adaptive_lr_num_steps = max(
            1, cfg.num_env_steps // self.num_envs - cfg.perception_iters * cfg.train_every)
        self.finetune_iters_completed = 0
        self._training_target_iter = None
        self._final_sac_active = False
        self._rl_steps_in_iteration = 0
        self._checkpoint_loaded = False
        self.needs_env_reset = False
        self._student = None
        self._student_rollout = Network(self.actor_adapt)
        self._adapt_buffer, self._adapt_t = None, 0
        self.actor.requires_grad_(False).eval()
        self.critic.requires_grad_(False)
        self.temperature.requires_grad_(False)
        self.vel_command_dim = observation_spec[VEL_CMD_KEY].shape[-1]
        self.student_replay_dim = self.replay_obs_dim + self.vel_command_dim + cfg.latent_dim
        layout = {k: (s, e) for k, s, e, _ in self.replay_layout}
        policy_start, policy_end = layout[OBS_KEY]
        self._student_replay_index = torch.tensor([
            *range(self.replay_obs_dim, self.replay_obs_dim + self.vel_command_dim),
            *range(policy_start, policy_end),
            *range(self.replay_obs_dim + self.vel_command_dim, self.student_replay_dim),
        ], device=self.device)
        if cfg.cycle_mode:
            print(f"[FlashSAC finetune] adaptive: teacher {cfg.perception_iters}, then direct SAC "
                  f"with reference from replay filling; SAC {cfg.cycle_sac_iters_min}..{cfg.cycle_sac_iters_max}, "
                  f"RMSE trigger +{cfg.perception_trigger:.1%}; refresh "
                  f"{cfg.cycle_perception_iters_min}..{cfg.cycle_perception_iters_max} iterations.")
        else:
            print(f"[FlashSAC finetune] first cycle: perception phase {cfg.perception_iters} + SAC "
                  f"{cfg.sac_iters} iterations; then {cfg.num_cycles} x (perception phase "
                  f"{cfg.cycle_perception_iters} + SAC {cfg.cycle_sac_iters}).")

    @property
    def is_repeated_perception_phase(self):
        return self.stage == 1 and (self._student_perception_phase or self.cycles_completed > 0)

    def _perception_length(self):
        if self.cfg.cycle_mode and getattr(self, "_student_perception_phase", False):
            return self.cfg.cycle_perception_iters_max
        if getattr(self, "_student_perception_phase", False):
            return self.cfg.cycle_perception_iters
        return self.cfg.perception_iters if self.cycles_completed == 0 else self.cfg.cycle_perception_iters

    def _rl_phase_length(self):
        if self.cfg.cycle_mode:
            return self.cfg.cycle_sac_iters_max
        return self.cfg.sac_iters if self.cycles_completed == 0 else self.cfg.cycle_sac_iters

    def configure_training_budget(self, total_iters):
        """Set this invocation's additional nominal budget before collecting data.

        A checkpoint already finishing its final SAC tail retains that pending
        intent and its original target. Ordinary resumes receive a new budget.
        """
        if isinstance(total_iters, bool) or not isinstance(total_iters, Integral) or total_iters < 0:
            raise ValueError("training budget must be a nonnegative number of iterations")
        if not (self.cfg.cycle_mode and self.cfg.final_sac_iters > 0 and self._final_sac_active):
            self._training_target_iter = self.finetune_iters_completed + int(total_iters)
            self._final_sac_active = False
        self._activate_final_sac_if_due()

    def _activate_final_sac_if_due(self):
        # Called only after complete iterations (or budget configuration), before
        # considering another SAC -> perception transition at the same boundary.
        if (self.cfg.cycle_mode and self.cfg.final_sac_iters > 0
                and self._training_target_iter is not None
                and self.finetune_iters_completed >= self._training_target_iter
                and not self._final_sac_active):
            self._final_sac_active = True
            print("[FlashSAC finetune] Nominal training budget reached; finishing the current "
                  "perception phase if needed, then requiring "
                  f"{self.cfg.final_sac_iters} uninterrupted SAC phase iterations.")

    def training_complete(self):
        """Whether the configured budget and any pending final SAC tail are done."""
        if self._training_target_iter is None or self.finetune_iters_completed < self._training_target_iter:
            return False
        if not self.cfg.cycle_mode or self.cfg.final_sac_iters == 0:
            return True
        return (self._final_sac_active and self.stage == 2
                and self.rl_phase_iters_completed >= self.cfg.final_sac_iters)

    def _final_sac_iters_remaining(self):
        if not self._final_sac_active:
            return 0
        completed = self.rl_phase_iters_completed if self.stage == 2 else 0
        return max(0, self.cfg.final_sac_iters - completed)

    def _rollout_noise_scale(self, teacher_rollout):
        if teacher_rollout:
            return self.cfg.teacher_perception_rollout_noise_scale
        if self.is_repeated_perception_phase:
            return self.cfg.student_perception_rollout_noise_scale
        return 1.0

    def _build_adaptation(self, observation_spec):
        super()._build_adaptation(observation_spec)
        # Keep the teacher checkpoint's parameter names/shapes; switch only the input keys.
        keys = [OBS_KEY]
        if self.cfg.adapt_module_input_cmd:
            keys.append(VEL_CMD_KEY)
        keys += [OBJECT_PRED_KEY, OBJECT_PRED_TRANS_KEY]
        self.adapt_module = Seq(
            CatTensors(keys, "_adapt_inp", del_keys=False, sort=False),
            self.adapt_module[1],
            selected_out_keys=[PRIV_PRED_KEY, ("next", "adapt_hx")],
        ).to(self.device)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)
        cnn = nn.Sequential(
            make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
            nn.LazyLinear(self.depth_feature_dim), nn.LayerNorm(self.depth_feature_dim),
        )
        self.temporal_depth_gru = TemporalDepthGRU(cnn, self.depth_feature_dim).to(self.device)
        fake = observation_spec.zero()[:2].to(self.device)
        fake["is_init"] = torch.ones(2, 1, dtype=torch.bool, device=self.device)
        fake["depth_hx"] = torch.zeros(2, self.depth_feature_dim, device=self.device)
        with torch.no_grad():
            self.temporal_depth_gru(fake)
        for module in cnn.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.temporal_depth_gru_ema = copy.deepcopy(self.temporal_depth_gru).requires_grad_(False)
        self.opt_adapt = optim.Adam(self._perception_parameters(), lr=self.cfg.adapt_learning_rate)

    def _perception_pairs(self):
        return ((self.temporal_depth_gru, self.temporal_depth_gru_ema),
                (self.object_adapt, self.object_adapt_ema),
                (self.adapt_module, self.adapt_ema))

    def _perception_parameters(self):
        return [p for model in (self.temporal_depth_gru, self.object_adapt, self.adapt_module)
                for p in model.parameters()]

    def make_tensordict_primer(self):
        return TensorDictPrimer({
            "adapt_hx": Unbounded((self.num_envs, self.cfg.latent_dim), device=self.device),
            "depth_hx": Unbounded((self.num_envs, self.depth_feature_dim), device=self.device),
        }, reset_key="done")

    def get_rollout_policy(self, mode="train"):
        return FinetuneRollout(self, mode)

    def _perceive(self, td, ema):
        depth, obj, adapt = (
            (self.temporal_depth_gru_ema, self.object_adapt_ema, self.adapt_ema) if ema
            else (self.temporal_depth_gru, self.object_adapt, self.adapt_module)
        )
        # PPOVEL's single-step GRU assumes its caller resets hidden states.
        # Also mask explicitly here for manual next-observation and evaluation calls.
        if td.ndim == 1:
            for key in ("depth_hx", "adapt_hx"):
                td[key] = torch.where(td["is_init"], 0., td[key])
        depth(td)
        obj(td)
        td[OBJECT_PRED_TRANS_KEY] = self.actor.transform_object_pose(td[OBJECT_PRED_KEY])
        adapt(td)
        return td

    def _joint_action(self, u, td):
        if self.cfg.action_mode == "residual":
            return td[REF_JPOS_KEY] + self.cfg.residual_scale * u
        return self.cfg.action_scale * u

    @torch.no_grad()
    def _record_perception_error(self, td, target=None):
        # Use the deployed EMA prediction that produced this action, before any
        # next-state perception/compiled call can reuse its output storage.
        if target is None:
            target = self.actor.encode_priv(self.replay_obs(td))
        error = (td[PRIV_PRED_KEY].float() - target.float()).square().mean(-1).reshape(-1)
        valid = td["step_count"].reshape(-1) > 1
        self._perception_error_sum.add_(torch.where(valid, error, 0.).sum(dtype=torch.float64))
        self._perception_error_count.add_(valid.sum())

    def _finish_perception_measurement(self):
        # Only synchronize once per complete iteration; no depth is kept in SAC
        # replay for monitoring. The controller retains just scalar sums/counts.
        if self._adaptive_schedule is not None:
            self._adaptive_schedule.observe(
                self._perception_error_sum.item(), self._perception_error_count.item())
        self._perception_error_sum.zero_()
        self._perception_error_count.zero_()
        self._perception_iteration_ready = False

    def _capture_schedule_info(self):
        schedule = self._adaptive_schedule
        if schedule is None:
            return
        self._last_schedule_info = {
            "finetune/measurement_stage": self.stage,
            "finetune/measurement_student_rollout": int(self.stage == 2 or self.is_repeated_perception_phase),
            **{f"finetune/{key}": value for key, value in {
                "perception_ema_rmse": schedule.last_error,
                "perception_ema_rmse_smoothed": schedule.smoothed_error,
                "relative_error": schedule.relative_error,
                "plateau_improvement": schedule.plateau_improvement,
                "trigger_streak": schedule.trigger_streak,
                "plateau_streak": schedule.plateau_streak,
            }.items() if value is not None},
        }

    @torch.no_grad()
    def add_transition(self, td):
        if self.stage == 1:
            keys = [OBS_KEY, VEL_CMD_KEY, DEPTH_KEY, OBJECT_KEY, REF_JPOS_KEY,
                    "is_init", "adapt_hx", "depth_hx"]
            step = td.select(*keys)
            # Teacher labels for _train_perception, computed per step (the teacher is frozen), so
            # neither the full replay observation nor one large teacher call is needed later.
            raw = self.replay_obs(td)
            step[PRIV_FEATURE_KEY] = self.actor.encode_priv(raw)
            if self.cfg.cycle_mode:
                self._record_perception_error(td, target=step[PRIV_FEATURE_KEY])
            mean, _ = self.actor.get_mean_and_std(raw, training=False)
            step["_teacher_action"] = self._joint_action(torch.tanh(mean), td)
            if self._adapt_t == 0:
                # One [num_envs, train_every] buffer per iteration, filled in place.
                n, t = step.shape[0], self.cfg.train_every
                self._adapt_buffer = TensorDict(
                    {k: torch.empty(n, t, *v.shape[1:], dtype=v.dtype, device=v.device)
                     for k, v in step.items()}, batch_size=[n, t], device=self.device)
            self._adapt_buffer[:, self._adapt_t] = step
            self._adapt_t += 1
            if self._adapt_t == self.cfg.train_every:
                self._pending_adapt, self._adapt_buffer, self._adapt_t = self._adapt_buffer, None, 0
                self._perception_iteration_ready = True
            return

        if self.cfg.cycle_mode:
            self._record_perception_error(td)
        next_td = td["next"]
        if ACTION_DR_KEY in self.replay_keys:
            next_td[ACTION_DR_KEY] = td[ACTION_DR_KEY]
        # Use the post-action observation and the hidden state produced at s_t. Do NOT
        # reset on done here: truncations bootstrap from the terminal, pre-reset state.
        perception_next = TensorDict({
            OBS_KEY: next_td[OBS_KEY], VEL_CMD_KEY: next_td[VEL_CMD_KEY],
            DEPTH_KEY: next_td[DEPTH_KEY],
            "is_init": torch.zeros_like(td["is_init"]),
            "adapt_hx": next_td["adapt_hx"], "depth_hx": next_td["depth_hx"],
        }, batch_size=td.batch_size, device=self.device)
        self._perceive(perception_next, ema=True)
        current_obs = self._student_replay_obs(td, td[PRIV_PRED_KEY])
        next_obs = self._student_replay_obs(next_td, perception_next[PRIV_PRED_KEY])
        if self.buffer is None:
            self._allocate_student_replay()
        terminated = next_td["terminated"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            time_limit = next_td["step_count"].squeeze(-1) >= self.env.max_episode_length
            cut = cut | (truncated & ~time_limit)
        reward = next_td["reward"].sum(-1)
        self.buffer.add(
            obs=current_obs if self.buffer.step == 0 else None, next_obs=next_obs,
            action=td[U_KEY], reward=reward, discount=next_td["discount"].squeeze(-1),
            done=terminated | truncated, cut=cut, valid=td["step_count"].squeeze(-1) > 1,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward, terminated, truncated)
        self._rl_steps_in_iteration += 1
        if self._rl_steps_in_iteration == self.cfg.train_every:
            self.finetune_iters_completed += 1
            self.rl_phase_iters_completed += 1
            self._rl_steps_in_iteration = 0
            self._perception_iteration_ready = True

    def _student_replay_obs(self, td, latent):
        return torch.cat([self.replay_obs(td), td[VEL_CMD_KEY], latent], dim=-1)

    def _allocate_student_replay(self):
        storage = "cpu" if self.cfg.buffer_device_type == "cpu" else self.device
        if torch.device(storage).type == "cuda":
            capacity = max(self.cfg.buffer_max_length // self.num_envs, self.cfg.n_step + 2) * self.num_envs
            itemsize = torch.finfo(getattr(torch, self.cfg.buffer_obs_dtype)).bits // 8
            required = capacity * (self.student_replay_dim * itemsize + self.action_dim * 4 + 15)
            free, _ = torch.cuda.mem_get_info(self.device)
            if required > free - 4 * 2**30:
                raise MemoryError("Student replay exceeds free GPU memory; use algo.buffer_device_type=cpu "
                                  "or reduce algo.buffer_max_length")
        self.buffer = NStepTrajectoryReplay(
            num_envs=self.num_envs, obs_dim=self.student_replay_dim, action_dim=self.action_dim,
            n_step=self.cfg.n_step, gamma=self.cfg.gamma, max_length=self.cfg.buffer_max_length,
            min_length=self.cfg.buffer_min_length, batch_size=self.cfg.sample_batch_size,
            device=self.device, obs_dtype=getattr(torch, self.cfg.buffer_obs_dtype), storage_device=storage,
        )
        print(f"[FlashSAC finetune] replay {self.buffer.capacity} rows, "
              f"{self.buffer.nbytes() / 2**30:.2f} GiB on {self.buffer.storage}")

    def update(self):
        if self.stage == 1:
            if self._pending_adapt is not None:
                info = self._train_perception(self._pending_adapt)
                self._pending_adapt = None
                for key, value in info.items():
                    self._info_sum[key] = self._info_sum.get(key, 0.) + value.detach()
                    self._info_cnt[key] = self._info_cnt.get(key, 0) + 1
                self.perception_iters_completed += 1
                self.finetune_iters_completed += 1
                self._finish_perception_measurement()
                self._activate_final_sac_if_due()
                if self.cfg.cycle_mode and self.is_repeated_perception_phase:
                    reason = self._adaptive_schedule.refresh_reason(self.perception_iters_completed)
                    if reason:
                        self._adaptive_schedule.establish_reference()
                        self._initial_reference_pending = False
                    self._capture_schedule_info()
                    if reason:
                        self._last_transition_reason = reason
                        self._start_rl()
                elif self.perception_iters_completed >= self._perception_length():
                    self._capture_schedule_info()
                    if self.cfg.cycle_mode:
                        self._last_transition_reason = "initial_sac"
                    self._start_rl()
                else:
                    self._capture_schedule_info()
            return
        if self._initial_reference_pending and self.can_start_training():
            # Capture the student error before the first eligible SAC update.
            # Replay may become ready partway through an iteration: include that
            # partial rollout without consuming it or postponing any updates.
            if self._adaptive_schedule.try_initialize_reference(
                    self._perception_error_sum.item(), self._perception_error_count.item()):
                self._initial_reference_pending = False
            # Missing/invalid observations leave the reference pending; SAC keeps
            # its original update timing and the next valid rollout is retried.
        FlashSACVelFlat.update(self)
        if not self._perception_iteration_ready:
            return
        self._finish_perception_measurement()
        self._activate_final_sac_if_due()
        if self.cfg.cycle_mode:
            reason = (None if self._final_sac_active
                      else self._adaptive_schedule.sac_reason(self.rl_phase_iters_completed))
            self._capture_schedule_info()
            if reason:
                self.cycles_completed += 1
                self._last_transition_reason = reason
                self._start_perception(student=True)
        elif (self.cycles_completed < self.cfg.num_cycles and
              self.rl_phase_iters_completed >= self._rl_phase_length()):
            self.cycles_completed += 1
            self._start_perception()

    @set_recurrent_mode(True)
    def _train_perception(self, rollout):
        # rollout already holds the teacher labels PRIV_FEATURE_KEY and _teacher_action.
        infos = []
        for _ in range(self.cfg.adapt_epochs):
            for batch in make_batch(rollout, self.cfg.adapt_num_minibatches, self.cfg.train_every):
                self._perceive(batch, ema=False)
                valid = (~batch["is_init"]).float()
                priv_loss = ((batch[PRIV_PRED_KEY] - batch[PRIV_FEATURE_KEY]).square() * valid).mean()
                object_loss = ((batch[OBJECT_PRED_KEY] - batch[OBJECT_KEY]).square() * valid).mean()
                priv_pred_norm = batch[PRIV_PRED_KEY].detach().norm(dim=-1).mean()
                depth_feature_norm = batch["_depth_feature"].detach().norm(dim=-1).mean()
                self.opt_adapt.zero_grad(set_to_none=True)
                (priv_loss + object_loss).backward()
                nn.utils.clip_grad_norm_(self._perception_parameters(), self.cfg.max_grad_norm)
                self.opt_adapt.step()
                # Distill on the rollout perception (EMA), which will be frozen for SAC.
                # This also keeps the actor loss completely separate from perception gradients.
                with torch.no_grad():
                    self._perceive(batch, ema=True)
                inputs = self._actor_adapt_input(batch).flatten(0, 1).detach()
                info = {
                    "adapt/priv_loss": priv_loss.detach(), "adapt/object_loss": object_loss.detach(),
                    "adapt/priv_feature_norm": batch[PRIV_FEATURE_KEY].norm(dim=-1).mean().detach(),
                    "adapt/priv_pred_norm": priv_pred_norm,
                    "adapt/depth_feature_norm": depth_feature_norm,
                }
                if self.is_repeated_perception_phase:
                    # The SAC actor stays frozen; only monitor its distance from the teacher.
                    with torch.no_grad():
                        student_mean, _ = self.actor_adapt.get_mean_and_std(inputs, training=False)
                        student_action = self._joint_action(
                            torch.tanh(student_mean).reshape(*batch.batch_size, self.action_dim), batch)
                        mse = ((student_action - batch["_teacher_action"]).square() * valid).mean()
                    info["adapt/teacher_student_action_rmse"] = mse.sqrt()
                    infos.append(info)
                    continue
                student_mean, _ = self.actor_adapt.get_mean_and_std(inputs, training=True)
                student_action = self._joint_action(
                    torch.tanh(student_mean).reshape(*batch.batch_size, self.action_dim), batch)
                adapt_loss = ((student_action - batch["_teacher_action"]).square() * valid).mean()
                self.opt_adapt_actor.zero_grad(set_to_none=True)
                adapt_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_adapt.parameters(), self.cfg.max_grad_norm)
                self.opt_adapt_actor.step()
                self._normalize_actor_adapt_parameters()
                info["adapt/adapt_loss"] = adapt_loss.detach()
                info["adapt/teacher_student_action_rmse"] = adapt_loss.detach().sqrt()
                infos.append(info)
        with torch.no_grad():
            for source, target in self._perception_pairs():
                for src, dst in zip(source.parameters(), target.parameters()):
                    dst.lerp_(src, self.cfg.perception_ema_tau)
        return {k: torch.stack([i[k] for i in infos]).mean() for k in infos[0]}

    def _start_rl(self, schedule_updates=None):
        for pair in self._perception_pairs():
            for model in pair:
                model.requires_grad_(False).eval()
        self.actor_adapt.requires_grad_(True)
        self.critic.requires_grad_(True)
        self.temperature.requires_grad_(True)
        self.stage = 2
        self._student_perception_phase = False
        self._initial_reference_pending = (
            self.cfg.cycle_mode and self._adaptive_schedule.reference_error is None)
        self.rl_phase_iters_completed = 0
        self._reset_phase_rollout()
        print("[FlashSAC finetune] Stage 2: perception and EMA frozen; student SAC enabled. "
              f"Resetting environments/hidden states; replay starts empty ({self._last_transition_reason}).")
        if self._student is not None:
            # Continue SAC optimizer/scheduler state after repeated supervised perception phase.
            return
        # Existing critic/temperature optimizers are fresh when loading a teacher. They
        # have not stepped during perception phase. Student SAC gets a separate fresh optimizer.
        optimizer = optim.Adam(self.actor_adapt.parameters(), lr=self.cfg.learning_rate_peak,
                               fused=self.device.type == "cuda")
        te, horizon = self.cfg.train_every, self.cfg.num_env_steps // self.num_envs
        if self.cfg.cycle_mode:
            # This upper bound does not predict future refresh durations. The LR
            # clock below uses known global frame progress, not this update count.
            steps = self._adaptive_lr_num_steps
        else:
            # Fixed schedule's existing SAC update budget.
            elapsed, steps = self.cfg.perception_iters * te, 0
            phase = self.cfg.sac_iters * te
            for _ in range(self.cfg.num_cycles):
                if elapsed + phase >= horizon:
                    break
                steps += phase
                elapsed += phase + self.cfg.cycle_perception_iters * te
                phase = self.cfg.cycle_sac_iters * te
            steps = max(1, steps + max(0, horizon - elapsed))
        updates = schedule_updates or max(1, int(steps * self.cfg.updates_per_interaction_step))
        self._rl_schedule_updates = updates

        def lr_lambda(num_steps):
            if self.cfg.cycle_mode:
                num_steps = self._adaptive_lr_num_steps
            schedule = ramp_cosine_decay_scheduler(
                init_value=self.cfg.learning_rate_init, peak_value=self.cfg.learning_rate_peak,
                end_value=self.cfg.learning_rate_end,
                ramp_steps=int(self.cfg.learning_rate_ramp_rate * num_steps),
                decay_steps=max(1, int(self.cfg.learning_rate_decay_rate * num_steps)),
            )
            if self.cfg.cycle_mode:
                return lambda _: schedule(max(0, (self.finetune_iters_completed
                                                  - self._adaptive_lr_start_iter) * te)) / self.cfg.learning_rate_peak
            return lambda step: schedule(step) / self.cfg.learning_rate_peak

        # Initialize schedules once; later cycles retain their SAC update progress. The critic
        # steps on every update, the actor and temperature every actor_update_period updates.
        actor_updates = max(1, updates // self.flash_cfg.actor_update_period)
        self._critic.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._critic.optimizer, lr_lambda=lr_lambda(updates))
        self._temperature.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._temperature.optimizer, lr_lambda=lr_lambda(actor_updates))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda(actor_updates))
        self._student = Network(
            self.actor_adapt, optimizer, scheduler,
            compile_network=self.cfg.use_compile, compile_mode=self.flash_cfg.compile_mode,
            use_weight_normalization=True,
        )
        if self.cfg.use_compile:
            self._student.network.get_mean_and_std = torch.compile(
                self._student.network.get_mean_and_std, mode=self.flash_cfg.compile_mode)
        self._student.normalize_parameters()
        self._update_step = 0

    def _reset_phase_rollout(self):
        self.buffer = None
        self._adapt_buffer, self._adapt_t = None, 0
        self._pending_adapt = None
        self._rl_steps_in_iteration = 0
        self._perception_iteration_ready = False
        self._perception_error_sum.zero_()
        self._perception_error_count.zero_()
        if self._adaptive_schedule is not None:
            self._adaptive_schedule.reset_phase()
        self._update_counter = 0.
        self._cur_noise_repeat_count.zero_()
        self._cur_noise_repeat_n.fill_(1)
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r.zero_()
        self.needs_env_reset = True
        if self.device.type == "cuda":
            # Return the finished phase's cached blocks so the simulator (PhysX) can use them.
            torch.cuda.empty_cache()

    def _start_perception(self, *, student=None):
        self.stage = 1
        self._student_perception_phase = self.cycles_completed > 0 if student is None else student
        self.perception_iters_completed = 0
        self._reset_phase_rollout()
        for online, ema in self._perception_pairs():
            online.requires_grad_(True).train()
            ema.requires_grad_(False).eval()
        # is_repeated_perception_phase reads self.stage, which is already 1 here.
        self.actor_adapt.requires_grad_(not self.is_repeated_perception_phase).train(
            not self.is_repeated_perception_phase)
        for model in (self.critic, self.temperature):
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
        # Same as the original perception phase: supervised optimizers start without Adam moments.
        for optimizer in (self.opt_adapt, self.opt_adapt_actor):
            optimizer.state.clear()
        kind = ("frozen-student-rollout perception-only" if self.is_repeated_perception_phase
                else "teacher-rollout")
        print(f"[FlashSAC finetune] Stage 1: restarting {kind} perception phase "
              f"after SAC cycle {self.cycles_completed}; replay released ({self._last_transition_reason}).")

    def _update_once(self):
        batch = self.buffer.sample()
        for key, actor_key in (("observation", "actor_observation"),
                               ("next_observation", "actor_next_observation")):
            obs = batch[key]
            batch[actor_key] = obs.index_select(-1, self._student_replay_index)
            batch[key] = self.critic_obs(obs)
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        info = _update_networks(
            batch=batch, actor=self._student, critic=self._critic, target_critic=self._target_critic,
            temperature=self._temperature, cfg=self.flash_cfg,
            do_actor_update=self._update_step % self.flash_cfg.actor_update_period == 0,
            device=self.device, grad_scaler=self._grad_scaler,
        )
        self._update_step += 1
        return info

    def pop_info(self):
        info = super().pop_info()
        info.update({"finetune/stage": self.stage,
                     "finetune/perception_iters_completed": self.perception_iters_completed,
                     "finetune/rl_phase_iters_completed": self.rl_phase_iters_completed,
                     "finetune/cycles_completed": self.cycles_completed,
                     "finetune/iters_completed": self.finetune_iters_completed,
                     "finetune/student_perception_phase": int(self.is_repeated_perception_phase)})
        if self.cfg.cycle_mode:
            info.update(self._last_schedule_info)
            info["finetune/transition_reason"] = self._last_transition_reason
            info["finetune/initial_reference_pending"] = int(self._initial_reference_pending)
            if self._adaptive_schedule.reference_error is not None:
                info["finetune/reference_error"] = self._adaptive_schedule.reference_error
        if self._training_target_iter is not None:
            info["finetune/final_sac_active"] = int(self._final_sac_active)
            info["finetune/final_sac_iters_remaining"] = self._final_sac_iters_remaining()
            info["finetune/extra_iters"] = max(0, self.finetune_iters_completed - self._training_target_iter)
        optimizer = (self.opt_adapt_actor if self.stage == 1 and not self.is_repeated_perception_phase
                     else self._student.optimizer if self._student is not None else None)
        info["actor/lr"] = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.
        return info

    def state_dict(self):
        state = super().state_dict()
        state["finetune_version"] = 9
        state["finetune_method"] = "adaptive_cycles_v2" if self.cfg.cycle_mode else "cycles_v2"
        state["cycle_mode"] = self.cfg.cycle_mode
        state["final_sac_iters"] = self.cfg.final_sac_iters
        state["training_target_iter"] = self._training_target_iter
        state["final_sac_active"] = self._final_sac_active
        state["student_perception_phase"] = self._student_perception_phase
        state["last_transition_reason"] = self._last_transition_reason
        state.update({key: getattr(self.cfg, key) for key in _ADAPTIVE_SCHEDULE_FIELDS})
        if self._adaptive_schedule is not None:
            state["adaptive_schedule"] = self._adaptive_schedule.state_dict()
            state["initial_reference_pending"] = self._initial_reference_pending
            state["adaptive_lr_start_iter"] = self._adaptive_lr_start_iter
            state["adaptive_lr_num_steps"] = self._adaptive_lr_num_steps
        state["finetune_stage"] = self.stage
        state["perception_iters_completed"] = self.perception_iters_completed
        state["finetune_iters_completed"] = self.finetune_iters_completed
        state["student_replay_dim"] = self.student_replay_dim
        state["student_replay_index"] = self._student_replay_index.cpu()
        state["perception_iters"] = self.cfg.perception_iters
        state["sac_iters"] = self.cfg.sac_iters
        state["num_cycles"] = self.cfg.num_cycles
        state["cycle_perception_iters"] = self.cfg.cycle_perception_iters
        state["cycle_sac_iters"] = self.cfg.cycle_sac_iters
        state["rl_phase_iters_completed"] = self.rl_phase_iters_completed
        state["cycles_completed"] = self.cycles_completed
        for name in ("temporal_depth_gru", "temporal_depth_gru_ema"):
            state[name] = getattr(self, name).state_dict()
        if self._student is not None:
            state["rl_schedule_updates"] = self._rl_schedule_updates
            state["student_optimizer"] = self._student.optimizer.state_dict()
            state["student_scheduler"] = self._student.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        resume = state_dict.get("finetune_version") in (6, 7, 8, 9)
        saved_adaptive = resume and state_dict.get("cycle_mode", False)
        if "finetune_version" in state_dict and not resume:
            raise ValueError("Unsupported finetune checkpoint version")
        if state_dict.get("actor_adapt_arch") != "flashsac_v1":
            raise ValueError("Finetune requires a teacher checkpoint with FlashSAC actor_adapt")
        if resume:
            adaptive_method = ("adaptive_cycles_v2" if state_dict["finetune_version"] >= 8
                               else "adaptive_cycles_v1")
            method = adaptive_method if saved_adaptive else "cycles_v2"
            if state_dict.get("finetune_method") != method:
                raise ValueError("Unsupported finetune method")
            if state_dict["finetune_stage"] not in (1, 2):
                raise ValueError("Invalid finetune stage in checkpoint")
            if (state_dict["student_replay_dim"] != self.student_replay_dim or
                    not torch.equal(state_dict["student_replay_index"].cpu(), self._student_replay_index.cpu())):
                raise ValueError("Student observation layout differs from the saved finetune checkpoint")
            self._warn_schedule_change(state_dict)
        # The parent's loader validates the teacher critic/replay/action layouts. On transfer,
        # omit teacher optimizer/counters: this is a new phase, not continued teacher training.
        transfer = dict(state_dict)
        if not resume:
            for key in list(transfer):
                if key.endswith(("_optimizer", "_scheduler")) or key in (
                    "opt_adapt", "opt_adapt_actor", "grad_scaler", "update_step", "last_iter"
                ):
                    transfer.pop(key)
        super().load_state_dict(transfer, strict=strict)
        if resume:
            for name in ("temporal_depth_gru", "temporal_depth_gru_ema"):
                getattr(self, name).load_state_dict(state_dict[name], strict=strict)
            self.finetune_iters_completed = state_dict["finetune_iters_completed"]
            self.cycles_completed = state_dict.get("cycles_completed", 0)
            saved_student_phase = state_dict.get("student_perception_phase", self.cycles_completed > 0)
            self._last_transition_reason = state_dict.get("last_transition_reason", "resume")
            if self.cfg.cycle_mode and saved_adaptive:
                # The invocation's frame budget may change on resume. Retain the
                # original LR clock just as fixed mode retains its update budget.
                self._adaptive_lr_start_iter = state_dict["adaptive_lr_start_iter"]
                self._adaptive_lr_num_steps = state_dict["adaptive_lr_num_steps"]
            # A repeated perception phase also retains the preceding SAC optimizer/scheduler.
            if "student_optimizer" in state_dict:
                self._start_rl(schedule_updates=state_dict.get("rl_schedule_updates"))
                self._student.optimizer.load_state_dict(state_dict["student_optimizer"])
                self._student.scheduler.load_state_dict(state_dict["student_scheduler"])
                for name, network in (("critic", self._critic), ("temperature", self._temperature)):
                    network.optimizer.load_state_dict(state_dict[f"{name}_optimizer"])
                    network.scheduler.load_state_dict(state_dict[f"{name}_scheduler"])
                self._update_step = state_dict["update_step"]
            elif state_dict["finetune_stage"] == 2:
                raise ValueError("SAC checkpoint is missing the student optimizer")
            if state_dict["finetune_stage"] == 1:
                # This also restores the initial student refresh, which has no
                # SAC optimizer yet and cannot be inferred from cycles_completed.
                self._start_perception(student=saved_student_phase)
                for name in ("opt_adapt", "opt_adapt_actor"):
                    if name in state_dict:
                        getattr(self, name).load_state_dict(state_dict[name])
            else:
                self._student_perception_phase = saved_student_phase
            self.perception_iters_completed = state_dict["perception_iters_completed"]
            self.rl_phase_iters_completed = state_dict.get("rl_phase_iters_completed", 0)
            if self.cfg.cycle_mode:
                if saved_adaptive:
                    self._adaptive_schedule.load_state_dict(state_dict["adaptive_schedule"])
                    # Version 8 can save SAC during replay filling, before the
                    # initial reference exists. Version 7 always established it
                    # in a separate refresh before starting SAC.
                    self._initial_reference_pending = state_dict.get("initial_reference_pending", False)
                    reference = self._adaptive_schedule.reference_error
                    if not isinstance(self._initial_reference_pending, bool):
                        raise ValueError("Invalid initial_reference_pending in checkpoint")
                    if self._initial_reference_pending and reference is not None:
                        raise ValueError("Pending initial reference is already set in checkpoint")
                    if self.stage == 2 and reference is None and not self._initial_reference_pending:
                        raise ValueError("Adaptive SAC checkpoint has no student reference error")
                    if any(state_dict.get(key) != getattr(self.cfg, key) for key in _ADAPTIVE_SCHEDULE_FIELDS):
                        # A new window/threshold must not inherit old patience decisions.
                        self._adaptive_schedule.reset_phase()
                    self._capture_schedule_info()
                elif self.stage == 2:
                    # A legacy SAC checkpoint establishes its first adaptive
                    # reference while refilling replay, without an extra phase.
                    self._initial_reference_pending = True
                    self._last_transition_reason = "initialize_reference"
        elif self.cfg.perception_iters == 0:
            if self.cfg.cycle_mode:
                self._last_transition_reason = "initial_sac"
            self._start_rl()
        self._training_target_iter = None
        self._final_sac_active = False
        if (resume and saved_adaptive and self.cfg.cycle_mode and self.cfg.final_sac_iters > 0
                and state_dict.get("finetune_version") == 9):
            active = state_dict.get("final_sac_active", False)
            if not isinstance(active, bool):
                raise ValueError("Invalid final_sac_active in checkpoint")
            if active:
                target = state_dict.get("training_target_iter")
                if (isinstance(target, bool) or not isinstance(target, Integral)
                        or not 0 <= target <= self.finetune_iters_completed):
                    raise ValueError("Invalid final SAC training target in checkpoint")
                self._training_target_iter = int(target)
                self._final_sac_active = True
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r.zero_()
        self._checkpoint_loaded = True
        # The training loop resets the simulator at startup. Replay/hidden state are intentionally
        # not checkpointed, and RL resumes by filling a fresh buffer with the warm-start actor.
        self.needs_env_reset = False
        self.env.set_progress(self.finetune_iters_completed)
        cycle_progress = (f"adaptive refreshes={self.cycles_completed}" if self.cfg.cycle_mode
                          else f"repeated cycles={self.cycles_completed}/{self.cfg.num_cycles}")
        print(f"[FlashSAC finetune] {'resumed' if resume else 'loaded teacher'}; "
              f"stage={self.stage}, perception phase={self.perception_iters_completed}/{self._perception_length()}, "
              f"{cycle_progress}")
        return []

    def _warn_schedule_change(self, state):
        # The cycle schedule comes from the config, so it may change on resume (e.g. more cycles).
        keys = ("cycle_mode", "final_sac_iters", "perception_iters", "sac_iters", "num_cycles",
                "cycle_perception_iters", "cycle_sac_iters", *_ADAPTIVE_SCHEDULE_FIELDS)
        changed = [f"{k} {state[k]} -> {getattr(self.cfg, k)}" for k in keys
                   if k in state and state[k] != getattr(self.cfg, k)]
        if changed:
            print("[FlashSAC finetune] WARNING: cycle schedule differs from the checkpoint; "
                  "continuing with the current config: " + ", ".join(changed))
