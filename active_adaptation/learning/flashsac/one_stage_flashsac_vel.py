"""Joint teacher/student FlashSAC with a supervised gated recurrent perception.

Teacher and student own half the rollout environments and share one twin critic.
The teacher encoder/input layout is inherited unchanged from FlashSACVel. A fresh
student perception predicts its latent, with an auxiliary object head. Only the
supervised objective trains perception; both actors learn with SAC. A weak,
scheduled pull additionally aligns encoder_priv toward detached student latents.

Replay stores uint8 depth once, sparse pre-observation belief checkpoints, and
ordered chunks. Old checkpoints plus short burn-in approximate current-model
history; no stored latent is treated as a current perception prediction.
"""

import math
from dataclasses import dataclass
from numbers import Integral
from typing import List

import torch
from torch import nn, optim
from hydra.core.config_store import ConfigStore
from tensordict.nn import TensorDictModuleBase
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import TensorDictPrimer

from ..ppo.common import ACTION_KEY, OBS_KEY
from ..ppo.ppo_vel import DEPTH_KEY, VEL_CMD_KEY, OBJECT_KEY, PRIV_PRED_KEY, REF_JPOS_KEY
from .flashsac_vel import FlashSACVel, FlashSACVelConfig
from .flashsac_vel_flat import ACTION_DR_KEY, U_KEY, FlashSACVelFlat, _raw
from .flashsac_upstream.agent import _sample_flashsac_actions
from .flashsac_upstream.network import FlashSACActor, FlashSACTemperature
from .flashsac_upstream.scheduler import ramp_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .one_stage_perception import GatedBeliefPerception
from .one_stage_replay import OneStageSequenceReplay
from .one_stage_update import update_joint_networks


BELIEF_KEY = "belief_hx"


@dataclass
class OneStageFlashSACVelConfig(FlashSACVelConfig):
    _target_: str = "active_adaptation.learning.flashsac.one_stage_flashsac_vel.OneStageFlashSACVel"
    name: str = "one_stage_flashsac_vel"
    in_keys: List[str] = (*FlashSACVelConfig.in_keys, DEPTH_KEY)
    enable_rtx: bool = False
    # Dynamic recurrent replay stays eager; optionally compile the SAC networks.
    use_compile: bool = False
    adapt_module: str = "gated_belief"
    enable_residual_distillation: bool = False
    belief_hidden_dim: int = 256
    checkpoint_interval: int = 16
    burn_in: int = 16
    sequence_length: int = 32
    depth_num_bins: int = 100
    perception_updates_per_iteration: int = 1
    priv_loss_coef: float = 1.0
    object_loss_coef: float = 1.0
    teacher_pull_enabled: bool = True
    teacher_pull_coef: float = 0.001
    # Count successful perception optimizer updates (restored on resume).
    teacher_pull_warmup_updates: int = 100
    teacher_pull_ramp_updates: int = 1000
    teacher_pull_max_grad_ratio: float = 0.1
    buffer_device_type: str = "cpu"
    buffer_min_length: int = 100_000


ConfigStore.instance().store("one_stage_flashsac_vel", node=OneStageFlashSACVelConfig, group="algo")
ConfigStore.instance().store("one_stage_flashsac_vel_train", node=OneStageFlashSACVelConfig, group="algo")


class OneStageRollout(TensorDictModuleBase):
    def __init__(self, policy, mode="train"):
        super().__init__()
        if mode not in ("train", "eval", "student", "teacher"):
            raise ValueError(f"unknown rollout mode {mode}")
        object.__setattr__(self, "policy", policy)
        self.mode = mode
        # Student inference needs no privileged teacher inputs or true object pose.
        self.in_keys = [OBS_KEY, VEL_CMD_KEY, DEPTH_KEY, "is_init", BELIEF_KEY]
        if mode in ("train", "teacher"):
            self.in_keys = list(dict.fromkeys([*self.in_keys, *policy.replay_env_keys]))
        if policy.cfg.action_mode == "residual" and REF_JPOS_KEY not in self.in_keys:
            self.in_keys.append(REF_JPOS_KEY)
        self.out_keys = [ACTION_KEY, U_KEY, PRIV_PRED_KEY, ("next", BELIEF_KEY)]

    @torch.no_grad()
    def forward(self, td):
        p = self.policy
        init = td["is_init"].bool()
        hidden = td.get(BELIEF_KEY)
        if hidden is None:
            hidden = torch.zeros(*td.batch_size, p.cfg.belief_hidden_dim, device=p.device)
        valid = p._valid_observation(td)
        hidden = torch.where(init | ~valid.unsqueeze(-1), 0., hidden)
        # Store the checkpoint before processing this observation.
        td[BELIEF_KEY] = hidden
        context = torch.cat([td[OBS_KEY], td[VEL_CMD_KEY]], -1)
        latent, _, next_hidden, _ = p.perception(td[DEPTH_KEY].float(), context, hidden, init, valid)
        td[PRIV_PRED_KEY] = latent
        td["next", BELIEF_KEY] = next_hidden
        student_input = torch.cat([td[VEL_CMD_KEY], td[OBS_KEY], latent], -1)
        if self.mode == "train":
            if td.shape[0] != p.num_envs:
                raise ValueError("training rollout must retain the fixed teacher/student env partition")
            if ACTION_DR_KEY in p.replay_keys:
                td[ACTION_DR_KEY] = p.action_dr()
            teacher_obs = p.replay_obs(td)
            half = p.num_envs // 2
            teacher_u = p._sample_actor(p._actor, teacher_obs[:half], "teacher", init[:half])
            student_u = p._sample_actor(p._student, student_input[half:], "student", init[half:])
            u = torch.cat([teacher_u, student_u])
        elif self.mode == "teacher":
            if ACTION_DR_KEY in p.replay_keys:
                td[ACTION_DR_KEY] = p.action_dr()
            mean, _ = p._actor.apply("get_mean_and_std", p.replay_obs(td), training=False)
            u = torch.tanh(mean)
        else:
            mean, _ = p._student.apply("get_mean_and_std", student_input, training=False)
            u = torch.tanh(mean)
        td[U_KEY] = u
        td[ACTION_KEY] = p._joint_action(u, td)
        return td


class OneStageFlashSACVel(FlashSACVel):
    checkpoint_version = 1

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        for key in ("checkpoint_interval", "sequence_length", "belief_hidden_dim",
                    "perception_updates_per_iteration", "train_every", "n_step",
                    "sample_batch_size", "buffer_max_length", "actor_update_period"):
            value = getattr(cfg, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(cfg.burn_in, bool) or not isinstance(cfg.burn_in, Integral) or cfg.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if not 1 <= cfg.depth_num_bins <= 255 or not isinstance(cfg.depth_num_bins, Integral):
            raise ValueError("uint8 replay needs depth_num_bins in [1,255]")
        if cfg.sample_batch_size % (2 * cfg.sequence_length):
            raise ValueError("sample_batch_size must be divisible by 2 * sequence_length")
        if observation_spec.shape[0] % 2:
            raise ValueError("joint training needs an even number of environments")
        if cfg.buffer_min_length < 0 or cfg.buffer_min_length > cfg.buffer_max_length:
            raise ValueError("buffer_min_length must be in [0,buffer_max_length]")
        if cfg.buffer_device_type != "cpu":
            raise ValueError("one-stage depth replay currently uses CPU storage")
        if cfg.buffer_obs_dtype not in ("float16", "float32"):
            raise ValueError("buffer_obs_dtype must be float16 or float32")
        for key in ("teacher_pull_warmup_updates", "teacher_pull_ramp_updates"):
            value = getattr(cfg, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("priv_loss_coef", "object_loss_coef", "updates_per_interaction_step",
                    "teacher_pull_coef", "teacher_pull_max_grad_ratio"):
            if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if cfg.priv_loss_coef <= 0:
            raise ValueError("priv_loss_coef must be positive to train the teacher-latent head")
        for key in (DEPTH_KEY, VEL_CMD_KEY, OBJECT_KEY):
            if observation_spec.get(key, None) is None:
                raise ValueError(f"one-stage FlashSAC requires observation {key!r}")
        self.depth_shape = tuple(observation_spec[DEPTH_KEY].shape[1:])
        if len(self.depth_shape) != 3 or self.depth_shape[0] != 1:
            raise ValueError("depth must have shape [num_envs,1,height,width]")
        # Verify the sensor and replay use the same quantization; do not silently
        # requantize an unquantized depth stream or a different sensor bin count.
        depth_group = getattr(env, "observation_funcs", {}).get(DEPTH_KEY)
        for sensor in getattr(depth_group, "funcs", {}).values():
            if hasattr(sensor, "num_bins") and sensor.num_bins != cfg.depth_num_bins:
                raise ValueError("depth sensor num_bins must equal algo.depth_num_bins")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self._interaction_steps = 0
        self._perception_version = 0
        self._perception_updates = 0
        self._noise = {}
        self.teacher_env_mask = torch.arange(self.num_envs, device=self.device) < self.num_envs // 2
        self.student_env_mask = ~self.teacher_env_mask
        self.replay_dim = self.replay_obs_dim + self.vel_command_dim
        self._sequence_batch_size = cfg.sample_batch_size // cfg.sequence_length

    def _make_network(self, module, weight_normalization=False):
        cfg = self.cfg
        optimizer = optim.Adam(module.parameters(), lr=cfg.learning_rate_peak,
                               fused=self.device.type == "cuda")
        schedule = ramp_cosine_decay_scheduler(
            init_value=cfg.learning_rate_init, peak_value=cfg.learning_rate_peak,
            end_value=cfg.learning_rate_end, ramp_steps=self.flash_cfg.learning_rate_ramp_step,
            decay_steps=self.flash_cfg.learning_rate_decay_step,
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: schedule(t) / cfg.learning_rate_peak)
        network = Network(module, optimizer, scheduler, compile_network=cfg.use_compile,
                          compile_mode=self.flash_cfg.compile_mode,
                          use_weight_normalization=weight_normalization)
        if weight_normalization:
            network.normalize_parameters()
            if cfg.use_compile:
                network.network.get_mean_and_std = torch.compile(
                    network.network.get_mean_and_std, mode=self.flash_cfg.compile_mode)
        return network

    def _build_adaptation(self, observation_spec):
        # Called by FlashSACVel after constructing its unchanged privileged actor.
        # Deliberately do not create the two legacy GRUs/object-transform adaptor.
        cfg = self.cfg
        self.vel_command_dim = observation_spec[VEL_CMD_KEY].shape[-1]
        self.policy_dim = observation_spec[OBS_KEY].shape[-1]
        geometry = self.actor.object_points_template
        if geometry is None:
            raise ValueError("gated belief requires known object-frame geometry")
        self.perception = GatedBeliefPerception(
            self.policy_dim + self.vel_command_dim, self.depth_shape,
            observation_spec[OBJECT_KEY].shape[-1], geometry,
            latent_dim=cfg.latent_dim, hidden_dim=cfg.belief_hidden_dim,
        ).to(self.device)
        self.opt_perception = optim.Adam(self.perception.parameters(), lr=cfg.adapt_learning_rate)
        self._student = self._make_network(FlashSACActor(
            cfg.actor_num_blocks, self.vel_command_dim + self.policy_dim + cfg.latent_dim,
            cfg.actor_hidden_dim, self.action_dim,
        ).to(self.device), weight_normalization=True)
        self.student_actor = _raw(self._student)
        self._student_temperature = self._make_network(
            FlashSACTemperature(cfg.temp_initial_value).to(self.device))
        self.student_temperature = _raw(self._student_temperature)

    def make_tensordict_primer(self):
        # The hidden spec already includes the environment batch dimension.
        # Explicit Composite shape works with both TorchRL 0.7 and 0.8+.
        spec = Composite({BELIEF_KEY: Unbounded(
            (self.num_envs, self.cfg.belief_hidden_dim), device=self.device)},
            shape=(self.num_envs,), device=self.device)
        return TensorDictPrimer(spec, reset_key="done")

    def get_rollout_policy(self, mode="train"):
        return OneStageRollout(self, mode)

    def _valid_observation(self, td):
        steps = td.get("step_count")
        if steps is not None:
            return steps.squeeze(-1) > 1
        return ~td["is_init"].squeeze(-1).bool()

    def _joint_action(self, u, td):
        if self.cfg.action_mode == "residual":
            return td[REF_JPOS_KEY] + self.cfg.residual_scale * u
        return self.cfg.action_scale * u

    def _sample_actor(self, actor, observation, role, is_init):
        if role not in self._noise:
            self._noise[role] = (
                torch.randn(observation.shape[0], self.action_dim, device=self.device),
                torch.zeros(observation.shape[0], 1, dtype=torch.int32, device=self.device),
                torch.ones(observation.shape[0], 1, dtype=torch.int32, device=self.device),
            )
        noise, count, repeats = self._noise[role]
        count = torch.where(is_init, 0, count)
        noise, action, count, repeats = _sample_flashsac_actions(
            actor, noise, observation, 1., count, repeats, self._zeta_cdf)
        self._noise[role] = (noise, count, repeats)
        return action

    def _stored_observation(self, td):
        return torch.cat([self.replay_obs(td), td[VEL_CMD_KEY]], -1)

    def _context(self, observation):
        return torch.cat([self.actor.field(observation[..., :self.replay_obs_dim], OBS_KEY),
                          observation[..., self.replay_obs_dim:]], -1)

    def _student_observation(self, observation, latent):
        return torch.cat([observation[..., self.replay_obs_dim:],
                          self.actor.field(observation[..., :self.replay_obs_dim], OBS_KEY), latent], -1)

    def _allocate_replay(self):
        cfg = self.cfg
        self.buffer = OneStageSequenceReplay(
            num_envs=self.num_envs, capacity=cfg.buffer_max_length,
            obs_dim=self.replay_dim, action_dim=self.action_dim, depth_shape=self.depth_shape,
            hidden_dim=cfg.belief_hidden_dim, checkpoint_interval=cfg.checkpoint_interval,
            burn_in=cfg.burn_in, sequence_length=cfg.sequence_length,
            n_step=cfg.n_step, gamma=cfg.gamma, num_depth_bins=cfg.depth_num_bins,
            obs_dtype=getattr(torch, cfg.buffer_obs_dtype), device=self.device,
            batch_size=self._sequence_batch_size, min_length=cfg.buffer_min_length,
        )
        print(f"[OneStage FlashSAC] replay {self.buffer.capacity:,} rows, "
              f"{self.buffer.nbytes() / 2**30:.2f} GiB; depth uint8, "
              f"hidden checkpoint every {cfg.checkpoint_interval} steps.")

    @torch.no_grad()
    def add_transition(self, td):
        nxt = td["next"]
        if ACTION_DR_KEY in self.replay_keys:
            nxt[ACTION_DR_KEY] = td[ACTION_DR_KEY]
        if self.buffer is None:
            self._allocate_replay()
        terminated = nxt["terminated"].squeeze(-1).bool()
        truncated = nxt["truncated"].squeeze(-1).bool()
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            time_limit = nxt["step_count"].squeeze(-1) >= self.env.max_episode_length
            cut = cut | (truncated & ~time_limit)
        reward = nxt["reward"].sum(-1)
        self.buffer.add(
            obs=self._stored_observation(td), depth=td[DEPTH_KEY], h_pre=td[BELIEF_KEY],
            is_init=td["is_init"].squeeze(-1), action=td[U_KEY], reward=reward,
            discount=nxt["discount"].squeeze(-1), done=terminated | truncated, cut=cut,
            valid=self._valid_observation(td), next_obs=self._stored_observation(nxt),
            next_depth=nxt[DEPTH_KEY], version=self._perception_version,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward, terminated, truncated)
        self._interaction_steps += 1

    def can_start_training(self):
        return self.buffer is not None and self.buffer.can_sample()

    def _record_info(self, info):
        for key, value in info.items():
            value = torch.as_tensor(value, device=self.device).detach().float()
            self._info_sum[key] = self._info_sum.get(key, 0.) + value
            self._info_cnt[key] = self._info_cnt.get(key, 0) + 1

    def update(self):
        if not self.can_start_training():
            return
        # Continuous joint learning: no perception-only or frozen-perception stage.
        # Keep perception fixed across the SAC updates between these boundaries.
        if self._perception_updates == 0 or self._interaction_steps % self.cfg.train_every == 0:
            for _ in range(self.cfg.perception_updates_per_iteration):
                self._record_info(self._train_perception(self.buffer.sample()))
        self._update_counter += self.cfg.updates_per_interaction_step
        while self._update_counter >= 1:
            self._record_info(self._update_once())
            self._update_counter -= 1

    @staticmethod
    def _gather(sequence, indices):
        rows = torch.arange(sequence.shape[0], device=sequence.device).unsqueeze(-1)
        return sequence[rows, indices]

    def _train_perception(self, batch):
        # Do not build a backward graph for the future bootstrap-only tail.
        end = int(batch.indices.max()) + 1
        obs = batch.obs[:, :end]
        z, obj, _, gates = self.perception.sequence(
            batch.depth[:, :end].float() / self.cfg.depth_num_bins, self._context(obs),
            batch.h0, batch.is_init[:, :end], batch.valid[:, :end], batch.learning_start,
        )
        z, obj = self._gather(z, batch.indices), self._gather(obj, batch.indices)
        learning_obs = self._gather(batch.obs, batch.indices)[..., :self.replay_obs_dim]
        with torch.no_grad():
            target = self.actor.encode_priv(learning_obs.flatten(0, 1)).reshape_as(z)
            object_target = self.actor.field(learning_obs, OBJECT_KEY)
        weights = batch.sample_weight
        priv_loss = ((z - target).square().mean(-1) * weights).sum()
        object_loss = ((obj - object_target).square().mean(-1) * weights).sum()
        loss = self.cfg.priv_loss_coef * priv_loss + self.cfg.object_loss_coef * object_loss
        self.opt_perception.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.perception.parameters(), self.cfg.max_grad_norm)
        self.opt_perception.step()
        # Leave no supervised gradients behind for the SAC gradient-isolation check.
        self.opt_perception.zero_grad(set_to_none=True)
        self._perception_version += 1
        self._perception_updates += 1
        gate = self._gather(gates, batch.indices).detach()
        return {
            "perception/loss": loss.detach(), "perception/priv_loss": priv_loss.detach(),
            "perception/object_loss": object_loss.detach(), "perception/grad_norm": norm.detach(),
            "perception/gate_mean": (gate.mean(-1) * weights).sum(),
            "perception/gate_closed_fraction": ((gate < .05).float().mean(-1) * weights).sum(),
            "perception/gate_open_fraction": ((gate > .95).float().mean(-1) * weights).sum(),
            "perception/checkpoint_age": (self._perception_version - batch.checkpoint_version).float().mean(),
        }

    @torch.no_grad()
    def _sac_batch(self, sequence):
        obs = sequence.obs
        z, _, hidden, _ = self.perception.sequence(
            sequence.depth.float() / self.cfg.depth_num_bins, self._context(obs),
            sequence.h0, sequence.is_init, sequence.valid,
        )
        current_obs = self._gather(obs, sequence.indices)
        current_z = self._gather(z, sequence.indices)
        next_obs = self._gather(obs, sequence.bootstrap_index).clone()
        next_z = self._gather(z, sequence.bootstrap_index).clone()
        terminal = sequence.terminal_mask & sequence.learning_mask
        if terminal.any():
            # Fork the true final observation from the belief after processing the
            # final action's state. The next episode's reset must not affect it.
            final_h = self._gather(hidden, sequence.bootstrap_index)[terminal]
            final_obs = sequence.terminal_obs[terminal]
            final_z, _, _, _ = self.perception(
                sequence.terminal_depth[terminal].float() / self.cfg.depth_num_bins,
                self._context(final_obs), final_h,
                torch.zeros(final_h.shape[0], 1, dtype=torch.bool, device=self.device),
            )
            next_obs[terminal] = final_obs
            next_z[terminal] = final_z
        # Episode lengths may change valid counts. Enforce 50:50 on the actual
        # transitions presented to SAC, not only on the number of sampled chunks.
        source = sequence.source[:, None].expand_as(sequence.learning_mask).flatten()
        valid = sequence.learning_mask.flatten()
        selected = []
        half = self.cfg.sample_batch_size // 2
        for role in (0, 1):
            candidates = torch.where(valid & (source == role))[0]
            if not candidates.numel():
                raise RuntimeError("sampled replay lacks valid transitions from one actor")
            selected.append(candidates[torch.randint(candidates.numel(), (half,), device=self.device)])
        index = torch.cat(selected)
        take = lambda value: value.flatten(0, 1).index_select(0, index)
        teacher_obs = current_obs[..., :self.replay_obs_dim]
        teacher_next_obs = next_obs[..., :self.replay_obs_dim]
        batch = {
            "observation": take(self.critic_obs(teacher_obs)),
            "next_observation": take(self.critic_obs(teacher_next_obs)),
            "actor_observation": take(teacher_obs), "actor_next_observation": take(teacher_next_obs),
            "student_observation": take(self._student_observation(current_obs, current_z)),
            "student_latent": take(current_z).detach(),
            "student_next_observation": take(self._student_observation(next_obs, next_z)),
            "action": take(sequence.actions),
            "reward": take(sequence.nstep_reward).squeeze(-1),
            "bootstrap_discount": take(sequence.bootstrap_discount).squeeze(-1),
        }
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        return batch

    def _teacher_pull_weight(self):
        if not self.cfg.teacher_pull_enabled:
            return 0.0
        elapsed = self._perception_updates - self.cfg.teacher_pull_warmup_updates
        if elapsed < 0:
            return 0.0
        ramp = self.cfg.teacher_pull_ramp_updates
        return self.cfg.teacher_pull_coef * (min(elapsed / ramp, 1.0) if ramp else 1.0)

    def _update_once(self):
        sequence = self.buffer.sample()
        batch = self._sac_batch(sequence)
        info = update_joint_networks(
            batch, self._actor, self._student, self._critic, self._target_critic,
            self._temperature, self._student_temperature, self.flash_cfg,
            self._update_step % self.cfg.actor_update_period == 0,
            self.device, self._grad_scaler,
            teacher_pull_coef=self._teacher_pull_weight(),
            teacher_pull_max_grad_ratio=self.cfg.teacher_pull_max_grad_ratio,
        )
        self._update_step += 1
        return info

    def pop_info(self):
        info = FlashSACVelFlat.pop_info(self)
        # teacher/* is reserved for teacher rollout tracking, parallel to train/*.
        info = {f"sac/{key}" if key.startswith(("teacher/", "student/")) else key: value
                for key, value in info.items()}
        info.update({"perception/version": self._perception_version,
                     "perception/updates": self._perception_updates,
                     "sac/student/actor/lr": self._student.optimizer.param_groups[0]["lr"]})
        info["sac/teacher/actor/pull_scheduled_coef"] = self._teacher_pull_weight()
        if self.buffer is not None:
            info["replay/bytes"] = self.buffer.nbytes()
        return info

    def _networks(self):
        return {**FlashSACVelFlat._networks(self), "student": self._student,
                "student_temperature": self._student_temperature}

    def _schema(self):
        return {
            "depth_shape": list(self.depth_shape), "depth_num_bins": self.cfg.depth_num_bins,
            "belief_hidden_dim": self.cfg.belief_hidden_dim, "latent_dim": self.cfg.latent_dim,
            "vel_command_dim": self.vel_command_dim, "checkpoint_interval": self.cfg.checkpoint_interval,
            "burn_in": self.cfg.burn_in, "sequence_length": self.cfg.sequence_length,
            "n_step": self.cfg.n_step, "gamma": self.cfg.gamma,
        }

    def state_dict(self):
        # The base serializer dispatches _networks(), including both actors and alphas.
        state = FlashSACVelFlat.state_dict(self)
        state["one_stage_version"] = self.checkpoint_version
        state["one_stage_schema"] = self._schema()
        state["replay_layout"] = [(k, s, e, list(shape)) for k, s, e, shape in self.replay_layout]
        state["perception"] = self.perception.state_dict()
        state["perception_optimizer"] = self.opt_perception.state_dict()
        state["perception_version"] = self._perception_version
        state["perception_updates"] = self._perception_updates
        state["interaction_steps"] = self._interaction_steps
        state["update_counter"] = self._update_counter
        return state

    def load_state_dict(self, state, strict=True):
        resume = "one_stage_version" in state
        if not resume and (self._interaction_steps or self._update_step or self._perception_updates):
            raise ValueError("load a teacher-only checkpoint into a fresh OneStageFlashSACVel instance")
        if resume and (state["one_stage_version"] != self.checkpoint_version
                       or state.get("one_stage_schema") != self._schema()):
            raise ValueError("one-stage checkpoint schema does not match the model/replay configuration")
        layout = [(k, s, e, list(shape)) for k, s, e, shape in self.replay_layout]
        if state.get("replay_layout") != layout or state.get("obs_layout") != self.obs_layout:
            raise ValueError("checkpoint teacher/critic observation layout does not match")
        if tuple(state.get("action_seam", ())) != self._action_seam():
            raise ValueError("checkpoint action convention does not match")
        for name in ("critic_state_encoder_dim", "critic_action_encoder_dim"):
            if state.get(name) != getattr(self.cfg, name):
                raise ValueError(f"checkpoint {name} does not match")
        if (self.cfg.critic_state_encoder_dim or self.cfg.critic_action_encoder_dim) and state.get(
                "critic_encoder_arch") != "relu_branches_v1":
            raise ValueError("checkpoint critic encoder architecture does not match")
        template = state["actor"].get("object_points_template")
        if template is None or not torch.equal(template.cpu(), self.actor.object_points_template.cpu()):
            raise ValueError("checkpoint object-frame geometry does not match this task")
        networks = self._networks() if resume else FlashSACVelFlat._networks(self)
        for name, network in networks.items():
            _raw(network).load_state_dict(state[name], strict=strict)
            if resume and network.optimizer is not None:
                network.optimizer.load_state_dict(state[f"{name}_optimizer"])
                network.scheduler.load_state_dict(state[f"{name}_scheduler"])
        if resume:
            self.perception.load_state_dict(state["perception"], strict=strict)
            self.opt_perception.load_state_dict(state["perception_optimizer"])
            self._perception_version = state["perception_version"]
            self._perception_updates = state["perception_updates"]
            self._interaction_steps = state["interaction_steps"]
            self._update_step = state["update_step"]
            self._update_counter = state.get("update_counter", 0.)
            self._grad_scaler.load_state_dict(state["grad_scaler"])
            self.env.set_progress(state.get("last_iter", 0))
        if self.reward_normalizer is not None and "reward_normalizer" in state:
            rn, saved = self.reward_normalizer, state["reward_normalizer"]
            # Replay and episodes restart; do not restore returns from old live envs.
            rn.G_r.zero_()
            rn.G_r_max = saved["G_r_max"].to(self.device)
            rn.G_rms.mean = saved["G_rms_mean"].to(self.device)
            rn.G_rms.var = saved["G_rms_var"].to(self.device)
            rn.G_rms.count = saved["G_rms_count"].to(self.device)
        self.buffer = None
        self._noise.clear()
        self._info_sum.clear()
        self._info_cnt.clear()
        print("[OneStage FlashSAC] " + ("resumed model/optimizers; replay starts empty" if resume
              else "loaded teacher/Q weights; student perception and optimizers start fresh"))
        return []
