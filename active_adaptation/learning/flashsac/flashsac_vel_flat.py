"""FlashSAC teacher using one flat actor/critic observation on VAIC tasks.

The network architecture and the update rule are FlashSAC's and come unmodified from the
vendored upstream code in `flashsac_upstream/`; rewards, observations, simulation, resets
and terminations are VAIC's. The two meet at the action: the actor emits u = tanh(z) in
[-1, 1] and the environment receives, depending on `action_mode`,

    action = ref_joint_pos_ + residual_scale * u     ("residual", VAIC's seam)
    action = action_scale * u                        ("absolute", as upstream FlashSAC)

where `ref_joint_pos_` is already in raw action coordinates. The actor and the critic both
read the same flat observation, which always ends with `ref_joint_pos_`.
"""
import contextlib
import functools
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union
from unittest import mock

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import II
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from termcolor import colored
from torch.amp.grad_scaler import GradScaler
from torchrl.envs.transforms import ObservationNorm, VecNorm

from ..ppo.common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY
from ..ppo.ppo_vel import (
    HEIGHT_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_TRANS_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    TransformObject,
)
from .flashsac_upstream import agent as flashsac_agent
from .flashsac_upstream.agent import (
    FlashSACConfig,
    _build_truncated_zeta_cdf,
    _init_flashsac_networks,
    _resolve_compile_mode,
    _sample_flashsac_actions,
    _update_networks,
)
from .flashsac_upstream.layer import EnsembleUnitBatchNorm, EnsembleUnitLinear
from .flashsac_upstream.network import FlashSACDoubleCritic
from .flashsac_upstream.reward_normalization import RewardNormalizer

U_KEY = "flashsac_u"  # normalized actor output u = tanh(z)
# privileged: this episode's JointPosition delay (physics substeps) and low-pass filter alpha
ACTION_DR_KEY = "action_dr"


@dataclass
class FlashSACVelFlatConfig:
    _target_: str = "active_adaptation.learning.flashsac.flashsac_vel_flat.FlashSACVelFlat"
    name: str = "flashsac_vel_flat"
    # FlashSAC normalizes its inputs with the BatchNorm embedder, so it consumes raw observations
    vecnorm: Union[str, None] = None
    # obs groups the env builds (same as ppo_vel_train) ...
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)
    # ... and the ones concatenated into the shared actor/critic input. Groups the task does not
    # define are skipped; ref_joint_pos_ is always appended. Add OBJECT_TRANS_KEY for the object
    # point cloud in the robot frame; ACTION_DR_KEY is the per-episode action delay and filter alpha.
    obs_keys: List[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY, OBJECT_KEY, HEIGHT_KEY, ACTION_DR_KEY)
    # "group/term" entries (term names as in the task's observation config) left out of the input;
    # by default the noisy copies in `policy` of values that `priv` holds without noise
    obs_drop_terms: List[str] = (
        "policy/root_ang_vel_history",
        "policy/projected_gravity_history",
        "policy/joint_pos_history",
    )
    # every future-reference term (term name contains "future") keeps only these of the task's
    # command.future_steps [1, 2, 8, 16, 32]; null keeps every step
    obs_future_steps: Optional[List[int]] = (8, 32)
    # null applies obs_future_steps to all future terms; a list of "group/term" limits it to those
    # (only needed to reload checkpoints with an older layout, see the README)
    obs_future_terms: Optional[List[str]] = None

    num_envs: int = 4096  # replaces task.num_envs; upstream scripts/run_isaaclab.sh num_train_envs
    train_every: int = 32  # env steps per train.py iteration (logging and checkpoint unit only)
    num_env_steps: int = II("oc.select:total_frames,-1")  # sets the learning-rate schedule length

    # How the actor output u = tanh(z) becomes the joint command:
    #   "residual" (VAIC's seam): action = ref_joint_pos_ + residual_scale * u, so the policy starts
    #       on the reference motion but can never leave a band of +-residual_scale around it.
    #   "absolute" (as upstream FlashSAC, which has no reference to center on):
    #       action = action_scale * u, no band, but no reference prior either.
    # ref_joint_pos_ stays in the observation in both modes.
    action_mode: str = "residual"
    residual_scale: float = 3.0
    # only used by "absolute". A trained ppo_vel_flat teacher on the skateboard task commands
    # |action| with p99 2.46 and |ref_joint_pos_| with p99 1.66, so 4.0 covers the working range
    # with headroom; a much larger scale spreads the same tanh range over a wider span and
    # coarsens the control, a smaller one reintroduces a cap.
    action_scale: float = 4.0
    # VAIC reports command completion as a truncation. The reference clip running out is an artificial
    # cut, not a failure, so the target bootstraps through it -- which is also what upstream FlashSAC
    # does (its target uses `terminated` alone) and what ppo_vel's GAE does (`1 - terminated`).
    # Setting this false makes FlashSAC solve a different MDP from the PPO baseline, since roughly
    # 60% of episodes on these tasks end this way.
    bootstrap_on_command_finished: bool = True

    # FlashSAC hyperparameters: configs/agent/flashSAC.yaml, with the scripts/run_isaaclab.sh
    # overrides for updates_per_interaction_step, n_step, buffer_max_length and buffer_min_length.
    updates_per_interaction_step: float = 8.0
    gamma: float = 0.99
    n_step: int = 3
    buffer_max_length: int = 10_000_000  # upstream IsaacLab uses 10_000_000; see the README
    buffer_obs_dtype: str = "float16"  # storage only; batches are cast back to float32
    # "cpu" keeps the replay in RAM (as upstream does for CPU simulators); batches still train on the GPU
    buffer_device_type: str = "cuda"
    # upstream's scripts/run_isaaclab.sh value. Until the buffer holds this many transitions the
    # actor is bypassed and actions are drawn uniformly, so this also sets how long the critic is
    # fit to nothing but random-action data before the actor starts following its gradient.
    buffer_min_length: int = 100_000
    sample_batch_size: int = 2048
    normalize_reward: bool = True
    normalized_G_max: float = 5.0

    learning_rate_init: float = 1e-4    #3e-4
    learning_rate_peak: float = 1e-4    #3e-4
    learning_rate_end: float = 1e-4     #1.5e-4
    learning_rate_ramp_rate: float = 1e-6
    learning_rate_decay_rate: float = 1.0

    actor_num_blocks: int = 2
    actor_hidden_dim: int = 256
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2

    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    # VAIC additions. Each Q head projects the state (BN -> unit linear -> ReLU) and action
    # (unit linear -> ReLU) before concatenation. The original FlashSAC embedder then applies
    # BN -> unit linear to the balanced features. Set both to null to restore the unmodified
    # upstream critic input.
    #   critic_state_encoder_dim shrinks the state so it does not outweigh the action by count.
    #   critic_action_encoder_dim expands the bounded action before fusion. Setting both encoder
    #     dimensions equally gives state and action equal representation in the original shared
    #     BatchNorm; without projection, the 1187-dim state can squeeze the 23 action inputs out
    #     of its learned scale budget (measured on a converged run: 0.15x their initial share).
    critic_state_encoder_dim: Optional[int] = 256
    critic_action_encoder_dim: Optional[int] = 256
    critic_num_bins: int = 101
    critic_target_update_tau: float = 0.01

    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.09 # flashsac: 0.15 but 0.09 was optimal for VAIC

    use_compile: bool = True
    compile_mode: str = "auto"
    use_amp: bool = True


cs = ConfigStore.instance()
cs.store("flashsac_vel_flat_train", node=FlashSACVelFlatConfig, group="algo")


# Sampling from RAM runs right after the updates wait on the GPU, when the intra-op thread pool
# has gone idle; waking all cores then costs ~6 ms per op on this machine, a few threads ~0.7 ms.
_CPU_SAMPLE_THREADS = 4


@contextlib.contextmanager
def _cpu_threads(num_threads: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _check_ram(nbytes: int, rows: int, reserve: int = 4 * 2**30):
    """Fail before allocating a CPU replay that would push the machine out of memory."""
    try:
        with open("/proc/meminfo") as f:
            available = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return
    if nbytes > available - reserve:
        fit = int(rows * (available - reserve) / nbytes)
        raise MemoryError(
            f"the CPU replay needs {nbytes / 2**30:.1f} GiB but {available / 2**30:.1f} GiB of RAM is available "
            f"(keeping {reserve / 2**30:.0f} GiB free); set algo.buffer_max_length to about {fit} or less"
        )


class NStepTrajectoryReplay:
    """Uniform n-step replay over `num_envs` parallel trajectories, stored on the GPU.

    Yields the same transitions as upstream `TorchUniformBuffer`: rows are drawn uniformly
    with replacement, and the n-step return stops at the first episode end and bootstraps
    from that step's next observation. Unlike upstream, each observation is stored once,
    in per-env time order, so the next observation of row t is the one in slot t+1.

    VAIC does not recompute observations on reset, so on the step after a reset the env
    carries a placeholder observation. That slot is filled with the true final observation
    of the finished episode instead, and the row itself is never sampled.

    Observations may be stored in a lower precision (`obs_dtype`); sampled batches are float32.
    With `storage_device="cpu"` the data lives in RAM and each batch is gathered into pinned
    memory and copied to `device` asynchronously.

    `include_one_step` also returns the first transition's next observation and continuation
    from the same sampled row. This lets an auxiliary one-step objective share the n-step
    reward replay without duplicating observation storage.
    """

    def __init__(
        self, num_envs, obs_dim, action_dim, n_step, gamma, max_length, min_length, batch_size, device,
        obs_dtype=torch.float32, storage_device=None, include_one_step=False,
    ):
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.min_length = min_length
        self.batch_size = batch_size
        self.include_one_step = include_one_step
        self.device = torch.device(device)  # where sampled batches are used
        self.storage = torch.device(storage_device) if storage_device is not None else self.device
        self.T = max(max_length // num_envs, n_step + 2)
        shape = (self.T, num_envs)
        if self.storage.type == "cpu":
            row_bytes = obs_dim * torch.finfo(obs_dtype).bits // 8 + action_dim * 4 + 15
            _check_ram(self.T * num_envs * row_bytes, num_envs * self.T)
        with torch.device(self.storage):
            self.obs = torch.zeros(*shape, obs_dim, dtype=obs_dtype)
            self.action = torch.zeros(*shape, action_dim)
            self.reward = torch.zeros(shape)
            self.discount = torch.ones(shape)
            self.done = torch.zeros(shape, dtype=torch.bool)
            self.cut = torch.zeros(shape, dtype=torch.bool)  # no bootstrap past this row
            self.valid = torch.zeros(shape, dtype=torch.bool)  # the row may start a transition
            self.weight = torch.zeros(shape)  # 1 where the row can be sampled right now
        self.step = 0  # global index of the next row
        if self.storage.type == "cpu" and self.device.type == "cuda":
            # reused pinned staging buffers and a separate stream, so copies do not queue behind the
            # training kernels and each batch lands in memory that is already mapped
            self._copy_stream = torch.cuda.Stream(self.device)
            obs_blocks = 3 if include_one_step else 2
            extra_scalars = 3 if include_one_step else 2
            self._staging = [
                (torch.empty(obs_blocks * batch_size, obs_dim, dtype=obs_dtype, pin_memory=True),
                 torch.empty(batch_size, action_dim + extra_scalars, pin_memory=True),
                 torch.cuda.Event())
                for _ in range(4)
            ]
            self._staging_index = 0

    @property
    def capacity(self) -> int:
        return self.T * self.num_envs

    def nbytes(self) -> int:
        tensors = (self.obs, self.action, self.reward, self.discount, self.done, self.cut, self.valid, self.weight)
        return sum(t.numel() * t.element_size() for t in tensors)

    def __len__(self) -> int:
        """Rows whose n-step window is complete, like upstream's count of emitted transitions."""
        ready_steps = min(self.step + 1 - self.n_step, self.T - self.n_step)
        return max(ready_steps, 0) * self.num_envs

    def can_sample(self) -> bool:
        return len(self) >= self.min_length

    def add(self, obs, next_obs, action, reward, discount, done, cut, valid):
        g, T = self.step, self.T
        s = g % T
        next_obs = next_obs.to(self.obs.dtype)  # cast before a possible copy to RAM
        if g == 0:
            self.obs[s] = obs.to(self.obs.dtype)
        self.action[s] = action
        self.reward[s] = reward
        self.discount[s] = discount
        self.done[s] = done
        self.cut[s] = cut
        self.valid[s] = valid
        self.weight[s] = 0.0
        # the next slot's previous occupant loses its observation
        self.obs[(g + 1) % T] = next_obs
        self.weight[(g + 1) % T] = 0.0
        j = g + 1 - self.n_step  # row whose window just became complete
        if j >= 0:
            self.weight[j % T] = self.valid[j % T].float()
        self.step += 1

    def sample(self):
        N = self.num_envs
        if self.storage.type != "cpu":
            # uniform with replacement over sampleable rows; much faster than torch.multinomial at this size
            cdf = torch.cumsum(self.weight.view(-1), 0)
            u = torch.rand(self.batch_size, device=cdf.device) * cdf[-1]
            idx = torch.searchsorted(cdf, u, right=True)
            return self.gather(idx // N, idx % N)
        # in RAM: the same rows (valid starts in the ready window), drawn by rejection, which is
        # cheaper on the CPU than a cumulative sum over every row
        with _cpu_threads(_CPU_SAMPLE_THREADS):
            last = self.step - 1
            first = max(0, last - self.T + 2)  # oldest row whose observation is still stored
            count = (last + 1 - self.n_step - first + 1) * N  # up to the newest row with a complete window
            idx = torch.randint(count, (self.batch_size,))
            t, env = (first + idx // N) % self.T, idx % N
            redo = ~self.valid[t, env]
            while redo.any():
                idx = torch.randint(count, (int(redo.sum()),))
                t[redo], env[redo] = (first + idx // N) % self.T, idx % N
                redo = ~self.valid[t, env]
            return self.gather(t, env)

    def gather(self, t: torch.Tensor, env: torch.Tensor):
        T, N, B = self.T, self.num_envs, t.shape[0]
        n_step_reward = torch.zeros(t.shape, device=t.device)
        scale = torch.ones_like(n_step_reward)  # product of the env's soft discounts so far
        cont = torch.ones_like(n_step_reward)
        alive = torch.ones_like(t, dtype=torch.bool)
        next_t = (t + self.n_step) % T
        for k in range(self.n_step):
            tk = (t + k) % T
            done = self.done[tk, env]
            n_step_reward += alive * (self.gamma**k) * scale * self.reward[tk, env]
            scale = torch.where(alive, scale * self.discount[tk, env], scale)
            ends = alive & done
            next_t = torch.where(ends, (tk + 1) % T, next_t)
            cont = torch.where(ends, scale * ~self.cut[tk, env], cont)
            alive = alive & ~done
        cont = torch.where(alive, scale, cont)

        row_blocks = [t * N + env, next_t * N + env]
        if self.include_one_step:
            row_blocks.append(((t + 1) % T) * N + env)
        rows = torch.cat(row_blocks)
        obs_rows = self.obs.view(-1, self.obs.shape[-1])
        # upstream's critic target multiplies the bootstrap by (1 - terminated)
        parts = [self.action.view(T * N, -1).index_select(0, rows[:B]), n_step_reward[:, None], 1.0 - cont[:, None]]
        if self.include_one_step:
            one_step_cont = self.discount[t, env] * ~(self.done[t, env] & self.cut[t, env])
            parts.append(1.0 - one_step_cont[:, None])
        if self.storage == self.device:
            obs, rest = obs_rows.index_select(0, rows), torch.cat(parts, dim=1)
        elif hasattr(self, "_staging") and B == self.batch_size:
            obs, rest = self._to_device(obs_rows, rows, parts)
        else:  # batches of other sizes (tests) take the plain path
            obs, rest = obs_rows.index_select(0, rows).to(self.device), torch.cat(parts, dim=1).to(self.device)
        obs = obs.float()
        A = self.action.shape[-1]
        batch = {
            "observation": obs[:B],
            "action": rest[:, :A].contiguous(),
            "reward": rest[:, A].contiguous(),
            "terminated": rest[:, A + 1].contiguous(),
            "next_observation": obs[B:2 * B],
        }
        if self.include_one_step:
            batch["one_step_next_observation"] = obs[2 * B:]
            batch["one_step_terminated"] = rest[:, A + 2].contiguous()
        return batch

    def _to_device(self, obs_rows, rows, parts):
        obs_pinned, rest_pinned, copied = self._staging[self._staging_index]
        self._staging_index = (self._staging_index + 1) % len(self._staging)
        copied.synchronize()  # the copy that last read this staging slot is done (normally long ago)
        torch.index_select(obs_rows, 0, rows, out=obs_pinned)
        torch.cat(parts, dim=1, out=rest_pinned)
        with torch.cuda.stream(self._copy_stream):
            obs = obs_pinned.to(self.device, non_blocking=True)
            rest = rest_pinned.to(self.device, non_blocking=True)
            copied.record(self._copy_stream)
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(copied)
        obs.record_stream(stream)
        rest.record_stream(stream)
        return obs, rest


class EncoderDoubleCritic(FlashSACDoubleCritic):
    """FlashSAC double critic with optional linear state/action projections before fusion.

    Upstream feeds `cat(state, action)` into one UnitBatchNorm whose scale and bias are
    renormalized to sqrt(d) after every update, so every input dimension draws on one fixed
    budget. With VAIC's 1187-dim state the 23 action inputs lose that competition: measured on
    a converged run, the action's share of the first layer had fallen to 0.15x its value at
    initialization (its BatchNorm scale 0.235 against the state's 0.761). Raising the action's
    share by shrinking the state alone does not survive training, because the two still share
    one budget. Projecting both branches to balanced feature counts removes that dimensional
    imbalance at the original FlashSAC fusion BatchNorm.

    State is normalized before its projection because it contains raw observations with mixed
    scales. Action is already bounded by tanh, so it is projected directly. ReLU after each
    projection turns the widened branches into nonlinear features; there is deliberately no
    branch-output BatchNorm because the original FlashSAC embedder immediately after concatenation
    supplies BN -> unit linear. The original residual trunk remains unchanged.
    """

    def __init__(
        self, num_blocks, input_dim, hidden_dim, num_bins, min_v, max_v, num_qs=2, *,
        state_dim, state_feature_dim=None, action_feature_dim=None,
    ):
        action_dim = input_dim - state_dim
        state_out = state_feature_dim or state_dim
        action_out = action_feature_dim or action_dim
        super().__init__(num_blocks, state_out + action_out, hidden_dim, num_bins, min_v, max_v, num_qs)
        self.state_norm = self.state_projection = None
        if state_feature_dim:
            self.state_norm = EnsembleUnitBatchNorm(num_qs, state_dim)
            self.state_projection = EnsembleUnitLinear(num_qs, state_dim, state_feature_dim)
        self.action_projection = None
        if action_feature_dim:
            self.action_projection = EnsembleUnitLinear(num_qs, action_dim, action_feature_dim)

    def forward(self, observations, actions, training):
        s = observations.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, state_dim]
        if self.state_projection is not None:
            s = torch.relu(self.state_projection(self.state_norm(s, training)))
        a = actions.unsqueeze(0).expand(self.num_qs, -1, -1)
        if self.action_projection is not None:
            a = torch.relu(self.action_projection(a))
        x = self.embedder(torch.cat((s, a), dim=-1), training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACRollout(TensorDictModuleBase):
    def __init__(self, policy: "FlashSACVelFlat", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.deterministic = mode != "train"
        self.in_keys = policy.env_obs_keys
        self.out_keys = [ACTION_KEY, U_KEY]

    def forward(self, tensordict: TensorDictBase):
        if ACTION_DR_KEY in self.policy.obs_keys:
            tensordict.set(ACTION_DR_KEY, self.policy.action_dr())
        u = self.policy.act(self.policy.flat_obs(tensordict), self.deterministic)
        tensordict.set(U_KEY, u)
        cfg = self.policy.cfg
        if cfg.action_mode == "residual":
            action = tensordict[REF_JPOS_KEY] + cfg.residual_scale * u
        else:
            action = cfg.action_scale * u
        tensordict.set(ACTION_KEY, action)
        return tensordict


def _raw(network):
    # `Network.network` is the torch.compile wrapper when compilation is on
    return getattr(network.network, "_orig_mod", network.network)


class FlashSACVelFlat(TensorDictModuleBase):
    is_off_policy = True

    def __init__(self, cfg: FlashSACVelFlatConfig, observation_spec, action_spec, reward_spec, device, env):
        super().__init__()
        self.cfg = cfg
        if cfg.action_mode not in ("residual", "absolute"):
            raise ValueError(f"algo.action_mode must be 'residual' or 'absolute', got {cfg.action_mode!r}")
        self.device = torch.device(device)
        object.__setattr__(self, "env", env)
        for transform in getattr(env.transform, "transforms", [env.transform]):
            if isinstance(transform, (VecNorm, ObservationNorm)):
                raise ValueError(
                    "FlashSAC normalizes its inputs with its BatchNorm embedder and expects raw "
                    "observations; run with vecnorm=null."
                )

        # shared actor/critic input
        spec_keys = set(observation_spec.keys())
        cmd_key = "command_" if "command_" in spec_keys else CMD_KEY
        self.obs_keys = []
        for key in cfg.obs_keys:
            key = cmd_key if key == CMD_KEY else key
            if key == OBJECT_TRANS_KEY:
                present = {OBJECT_KEY, OBJECT_GEO_KEY} <= spec_keys
            elif key == ACTION_DR_KEY:
                present = all(hasattr(env.action_manager, a) for a in ("delay", "alpha"))
            else:
                present = key in spec_keys
            if present:
                self.obs_keys.append(key)
            else:
                print(colored(f"[FlashSAC] obs group '{key}' is not defined by this task, skipped.", "yellow"))
        assert REF_JPOS_KEY in spec_keys, f"{REF_JPOS_KEY} is required for the residual action"
        self.obs_keys.append(REF_JPOS_KEY)
        self.env_obs_keys = []
        for key in self.obs_keys:
            if key == OBJECT_TRANS_KEY:
                self.env_obs_keys += [OBJECT_KEY, OBJECT_GEO_KEY]
            elif key != ACTION_DR_KEY:
                self.env_obs_keys.append(key)
        self.object_transform = TransformObject(None, [OBJECT_KEY, OBJECT_GEO_KEY], [OBJECT_TRANS_KEY])

        self.action_dim = action_spec.shape[-1]
        self.num_envs = observation_spec.shape[0]
        rename = lambda name: cmd_key + name[len(CMD_KEY):] if name.startswith(f"{CMD_KEY}/") else name
        future_terms = None if cfg.obs_future_terms is None else {rename(n) for n in cfg.obs_future_terms}
        index, self.obs_layout, num_future = self._select_terms(
            env, observation_spec, {rename(n) for n in cfg.obs_drop_terms}, future_terms
        )
        self._obs_index = torch.tensor(index, device=self.device)
        self.obs_dim = len(index)
        print(colored(f"[FlashSAC] actor/critic observation dim {self.obs_dim}:", "green"))
        for line in self.obs_layout:
            print(colored(f"    {line}", "green"))
        if num_future:
            print(colored(f"    {num_future} future terms keep future steps {list(cfg.obs_future_steps)}", "green"))

        # FlashSAC agent state, as built by upstream FlashSACAgent.__init__
        num_interaction_steps = max(1, cfg.num_env_steps // self.num_envs)
        num_updates = num_interaction_steps * cfg.updates_per_interaction_step
        self.flash_cfg = FlashSACConfig(
            seed=0,  # unused by the agent code
            normalize_reward=cfg.normalize_reward,
            normalized_G_max=cfg.normalized_G_max,
            asymmetric_observation=False,
            device_type=str(self.device),
            buffer_max_length=cfg.buffer_max_length,
            buffer_min_length=cfg.buffer_min_length,
            buffer_device_type=cfg.buffer_device_type,
            sample_batch_size=cfg.sample_batch_size,
            learning_rate_init=cfg.learning_rate_init,
            learning_rate_peak=cfg.learning_rate_peak,
            learning_rate_end=cfg.learning_rate_end,
            learning_rate_ramp_rate=cfg.learning_rate_ramp_rate,
            learning_rate_ramp_step=int(cfg.learning_rate_ramp_rate * num_updates),
            learning_rate_decay_rate=cfg.learning_rate_decay_rate,
            learning_rate_decay_step=int(cfg.learning_rate_decay_rate * num_updates),
            actor_num_blocks=cfg.actor_num_blocks,
            actor_hidden_dim=cfg.actor_hidden_dim,
            actor_bc_alpha=cfg.actor_bc_alpha,
            actor_noise_zeta_mu=cfg.actor_noise_zeta_mu,
            actor_noise_zeta_max=cfg.actor_noise_zeta_max,
            actor_update_period=cfg.actor_update_period,
            critic_num_blocks=cfg.critic_num_blocks,
            critic_hidden_dim=cfg.critic_hidden_dim,
            critic_num_bins=cfg.critic_num_bins,
            critic_min_v=-cfg.normalized_G_max,
            critic_max_v=cfg.normalized_G_max,
            critic_target_update_tau=cfg.critic_target_update_tau,
            temp_initial_value=cfg.temp_initial_value,
            temp_target_sigma=cfg.temp_target_sigma,
            temp_target_entropy=0.5 * self.action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2),
            gamma=cfg.gamma,
            n_step=cfg.n_step,
            use_compile=cfg.use_compile,
            compile_mode=_resolve_compile_mode(cfg.compile_mode),
            use_amp=cfg.use_amp,
            load_optimizer=True,
            load_reward_normalizer=True,
        )
        critic_cls = FlashSACDoubleCritic
        if cfg.critic_state_encoder_dim or cfg.critic_action_encoder_dim:
            critic_cls = functools.partial(
                EncoderDoubleCritic,
                state_dim=self.obs_dim,
                state_feature_dim=cfg.critic_state_encoder_dim,
                action_feature_dim=cfg.critic_action_encoder_dim,
            )
        # upstream wires the optimizers, schedules, weight normalization, target EMA and compilation;
        # only the critic class it instantiates is swapped
        with mock.patch.object(flashsac_agent, "FlashSACDoubleCritic", critic_cls):
            self._actor, self._critic, self._target_critic, self._temperature = _init_flashsac_networks(
                actor_observation_dim=self.obs_dim,
                critic_observation_dim=self.obs_dim,
                action_dim=self.action_dim,
                cfg=self.flash_cfg,
                device=self.device,
            )
        self.actor = _raw(self._actor)
        self.critic = _raw(self._critic)
        self.target_critic = _raw(self._target_critic)
        self.temperature = _raw(self._temperature)
        self._grad_scaler = GradScaler(device=self.device.type, enabled=cfg.use_amp)

        self._zeta_cdf = _build_truncated_zeta_cdf(mu=cfg.actor_noise_zeta_mu, max_n=cfg.actor_noise_zeta_max).to(self.device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self.device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self.device)
        self._cached_noise = torch.randn(self.num_envs, self.action_dim, device=self.device)

        self.reward_normalizer = None
        if cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=cfg.gamma, G_max=cfg.normalized_G_max, load_rms=True, device=self.device
            )

        self.buffer = None  # allocated on the first transition, so evaluation never pays for it
        self._update_step = 0
        self._update_counter = 0.0
        self._info_sum = {}
        self._info_cnt = {}

    def _select_terms(self, env, observation_spec, drop_terms, future_terms):
        """Indices of the kept entries in the concatenation of `self.obs_keys`, a readable layout, and
        the number of future terms reduced to `obs_future_steps`."""
        future_steps = self.cfg.obs_future_steps
        if future_terms is None:
            is_future = lambda name: "future" in name.split("/", 1)[1]
        else:
            is_future = lambda name: name in future_terms
        index, layout, offset, matched, num_future = [], [], 0, set(), 0
        for group in self.obs_keys:
            if group == OBJECT_TRANS_KEY:
                terms = [(OBJECT_TRANS_KEY, math.prod(observation_spec[OBJECT_GEO_KEY].shape[1:]))]
            elif group == ACTION_DR_KEY:
                terms = [("delay", 1), ("alpha", 1)]
            else:
                terms = [(t, f().reshape(self.num_envs, -1).shape[-1]) for t, f in env.observation_funcs[group].funcs.items()]
                assert sum(d for _, d in terms) == math.prod(observation_spec[group].shape[1:]), group
            group_dim, notes = sum(d for _, d in terms), []
            for term, dim in terms:
                name = f"{group}/{term}"
                keep = range(dim)
                if name in drop_terms:
                    keep = range(0)
                    notes.append(f"-{term} ({dim})")
                elif future_steps and is_future(name):
                    all_steps = env.command_manager.future_steps.tolist()
                    missing = set(future_steps) - set(all_steps)
                    if missing or dim % len(all_steps):
                        raise ValueError(f"cannot keep future steps {list(future_steps)} of {name} (task has {all_steps})")
                    per_step = dim // len(all_steps)  # future terms are laid out step-major
                    keep = [all_steps.index(s) * per_step + i for s in future_steps for i in range(per_step)]
                    notes.append(f"{term} {len(keep)}/{dim}")
                    num_future += 1
                matched.add(name)
                index.extend(offset + i for i in keep)
                offset += dim
            kept = sum(1 for i in index if i >= offset - group_dim)
            layout.append(f"{group} {kept}/{group_dim}" + (f": {', '.join(notes)}" if notes else ""))
        # entries naming a term of a used group that does not exist are typos
        named = drop_terms | (future_terms or set())
        unknown = sorted(n for n in named if n.split("/")[0] in self.obs_keys and n not in matched)
        if unknown:
            raise ValueError(f"unknown observation terms in obs_drop_terms/obs_future_terms: {unknown}")
        return index, layout, num_future

    def get_rollout_policy(self, mode: str = "train"):
        return FlashSACRollout(self, mode)

    def action_dr(self) -> torch.Tensor:
        action_manager = self.env.action_manager
        return torch.cat([action_manager.delay.float(), action_manager.alpha.float()], dim=-1)

    def flat_obs(self, tensordict: TensorDictBase) -> torch.Tensor:
        if OBJECT_TRANS_KEY in self.obs_keys:
            self.object_transform(tensordict)
        batch_size = tensordict.batch_size
        obs = torch.cat([tensordict[k].reshape(*batch_size, -1).float() for k in self.obs_keys], dim=-1)
        return obs.index_select(-1, self._obs_index)

    def can_start_training(self) -> bool:
        return self.buffer is not None and self.buffer.can_sample()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            mean, _ = self.actor.get_mean_and_std(obs, training=False)
            return torch.tanh(mean)
        if not self.can_start_training():
            # upstream samples uniform random actions until the buffer is warm
            return torch.rand(obs.shape[0], self.action_dim, device=obs.device) * 2 - 1
        (
            self._cached_noise,
            actions,
            self._cur_noise_repeat_count,
            self._cur_noise_repeat_n,
        ) = _sample_flashsac_actions(
            actor=self._actor,
            noise=self._cached_noise,
            observations=obs,
            temperature=1.0,
            cur_count=self._cur_noise_repeat_count,
            cur_n=self._cur_noise_repeat_n,
            zeta_cdf=self._zeta_cdf,
        )
        return actions

    @torch.no_grad()
    def add_transition(self, tensordict: TensorDictBase):
        """Store the output of `env.step_and_maybe_reset` (before its reset) in the replay buffer."""
        next_td = tensordict["next"]
        if self.buffer is None:
            self.buffer = NStepTrajectoryReplay(
                num_envs=tensordict.shape[0],
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_step=self.cfg.n_step,
                gamma=self.cfg.gamma,
                max_length=self.cfg.buffer_max_length,
                min_length=self.cfg.buffer_min_length,
                batch_size=self.cfg.sample_batch_size,
                device=self.device,
                obs_dtype=getattr(torch, self.cfg.buffer_obs_dtype),
                storage_device="cpu" if self.cfg.buffer_device_type == "cpu" else self.device,
            )
            print(colored(
                f"[FlashSAC] replay buffer: {self.buffer.capacity} rows "
                f"({self.buffer.T} steps x {self.buffer.num_envs} envs), "
                f"{self.buffer.nbytes() / 2**30:.2f} GiB on {self.buffer.storage}",
                "green",
            ))
        if ACTION_DR_KEY in self.obs_keys:
            # the next observation belongs to the same episode even if the env has reset since
            next_td.set(ACTION_DR_KEY, tensordict[ACTION_DR_KEY])
        reward = next_td["reward"].sum(-1)  # reward groups summed, as PPO sums their advantages
        terminated = next_td["terminated"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        time_limit = next_td["step_count"].squeeze(-1) >= self.env.max_episode_length
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            cut = cut | (truncated & ~time_limit)
        self.buffer.add(
            obs=self.flat_obs(tensordict) if self.buffer.step == 0 else None,
            next_obs=self.flat_obs(next_td),
            action=tensordict[U_KEY],
            reward=reward,
            discount=next_td["discount"].squeeze(-1),
            done=terminated | truncated,
            cut=cut,
            # the first two steps of an episode are invalid, as in PPOVEL (step_count > 1)
            valid=tensordict["step_count"].squeeze(-1) > 1,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward=reward, terminated=terminated, truncated=truncated)

    def update(self):
        """Run the gradient updates owed after one env step (upstream train.py loop)."""
        if not self.can_start_training():
            return
        self._update_counter += self.cfg.updates_per_interaction_step
        while self._update_counter >= 1:
            for key, value in self._update_once().items():
                self._info_sum[key] = self._info_sum.get(key, 0.0) + value.detach().float()
                self._info_cnt[key] = self._info_cnt.get(key, 0) + 1
            self._update_counter -= 1

    def _update_once(self):
        # upstream FlashSACAgent.update
        batch = self.buffer.sample()
        batch["actor_observation"] = batch["observation"]
        batch["actor_next_observation"] = batch["next_observation"]
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self.flash_cfg,
            do_actor_update=(self._update_step % self.flash_cfg.actor_update_period == 0),
            device=self.device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1
        return update_info

    def pop_info(self):
        info = {k: (v / self._info_cnt[k]).item() for k, v in sorted(self._info_sum.items())}
        self._info_sum.clear()
        self._info_cnt.clear()
        info["flashsac/update_step"] = self._update_step
        if self.buffer is not None:
            info["flashsac/buffer_rows"] = len(self.buffer)
        info["actor/lr"] = self._actor.optimizer.param_groups[0]["lr"]
        info["critic/lr"] = self._critic.optimizer.param_groups[0]["lr"]
        if self.reward_normalizer is not None:
            info["flashsac/G_r_max"] = self.reward_normalizer.G_r_max.item()
            info["flashsac/G_std"] = self.reward_normalizer.G_rms.var.sqrt().item()
        return info

    def _networks(self):
        return {
            "actor": self._actor,
            "critic": self._critic,
            "target_critic": self._target_critic,
            "temperature": self._temperature,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, network in self._networks().items():
            state_dict[name] = _raw(network).state_dict()
            if network.optimizer is not None:
                state_dict[f"{name}_optimizer"] = network.optimizer.state_dict()
                state_dict[f"{name}_scheduler"] = network.scheduler.state_dict()
        if self.reward_normalizer is not None:
            rn = self.reward_normalizer
            state_dict["reward_normalizer"] = {
                "G_r": rn.G_r, "G_r_max": rn.G_r_max,
                "G_rms_mean": rn.G_rms.mean, "G_rms_var": rn.G_rms.var, "G_rms_count": rn.G_rms.count,
            }
        state_dict["update_step"] = self._update_step
        state_dict["grad_scaler"] = self._grad_scaler.state_dict()
        state_dict["last_iter"] = self.env.current_iter
        state_dict["obs_layout"] = self.obs_layout
        state_dict["critic_state_encoder_dim"] = self.cfg.critic_state_encoder_dim
        state_dict["critic_action_encoder_dim"] = self.cfg.critic_action_encoder_dim
        state_dict["critic_encoder_arch"] = "relu_branches_v1"
        state_dict["action_seam"] = self._action_seam()
        return state_dict

    def _action_seam(self):
        """How u = tanh(z) is turned into the joint command; a checkpoint is only valid for its own."""
        if self.cfg.action_mode == "residual":
            return ("residual", float(self.cfg.residual_scale))
        return ("absolute", float(self.cfg.action_scale))

    def load_state_dict(self, state_dict, strict=True):
        saved_dim = state_dict["actor"]["embedder.w.w.weight"].shape[1]
        saved_layout = state_dict.get("obs_layout")
        if saved_layout is not None:  # older checkpoints also stored an informational "future steps" line
            saved_layout = [line for line in saved_layout if not line.startswith("future steps")]
        if saved_dim != self.obs_dim or (saved_layout is not None and saved_layout != self.obs_layout):
            raise ValueError(
                f"checkpoint input is {saved_dim}-dim ({saved_layout or 'layout not recorded'}) but the current "
                f"obs options give {self.obs_dim} ({self.obs_layout}). Set algo.obs_keys, obs_drop_terms, "
                "obs_future_steps and obs_future_terms as in the run's cfg.yaml; the README lists the overrides "
                "for earlier layouts."
            )
        saved_seam = state_dict.get("action_seam")  # absent in checkpoints from before action_mode
        if saved_seam is not None and tuple(saved_seam) != self._action_seam():
            mode, scale = saved_seam
            raise ValueError(
                f"checkpoint actor was trained with action_mode={mode} at scale {scale}, the current "
                f"settings give {self._action_seam()[0]} at {self._action_seam()[1]}; set "
                f"algo.action_mode={mode} algo.{'residual_scale' if mode == 'residual' else 'action_scale'}={scale}."
            )
        for option in ("critic_state_encoder_dim", "critic_action_encoder_dim"):
            saved = state_dict.get(option)  # absent in checkpoints from before the option
            if saved != getattr(self.cfg, option):
                raise ValueError(
                    f"checkpoint critic was trained with {option}={saved}, the current option is "
                    f"{getattr(self.cfg, option)}; set algo.{option}={'null' if saved is None else saved}."
                )
        uses_encoder = self.cfg.critic_state_encoder_dim or self.cfg.critic_action_encoder_dim
        saved_encoder_arch = state_dict.get("critic_encoder_arch")
        if uses_encoder and saved_encoder_arch != "relu_branches_v1":
            raise ValueError(
                f"checkpoint critic encoder architecture is {saved_encoder_arch or 'legacy'}, but the current "
                "critic uses relu_branches_v1 (state: BN/Linear/ReLU; action: Linear/ReLU). These architectures "
                "are not checkpoint-compatible; start a fresh run or use a checkpoint created with the current code."
            )
        # in-place loads keep the parameter tensors that the compiled EMA and weight-norm functions hold
        for name, network in self._networks().items():
            _raw(network).load_state_dict(state_dict[name], strict=strict)
            if network.optimizer is not None and f"{name}_optimizer" in state_dict:
                network.optimizer.load_state_dict(state_dict[f"{name}_optimizer"])
                network.scheduler.load_state_dict(state_dict[f"{name}_scheduler"])
        if self.reward_normalizer is not None and "reward_normalizer" in state_dict:
            rn, saved = self.reward_normalizer, state_dict["reward_normalizer"]
            if saved["G_r"].numel() == self.num_envs:  # per-env running returns
                rn.G_r = saved["G_r"].to(self.device)
            rn.G_r_max = saved["G_r_max"].to(self.device)
            rn.G_rms.mean = saved["G_rms_mean"].to(self.device)
            rn.G_rms.var = saved["G_rms_var"].to(self.device)
            rn.G_rms.count = saved["G_rms_count"].to(self.device)
        self._update_step = state_dict.get("update_step", 0)
        if "grad_scaler" in state_dict:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])
        self.env.set_progress(state_dict.get("last_iter", 0))
        print(colored(f"[FlashSAC] loaded checkpoint at update step {self._update_step}.", "green"))
        return []
