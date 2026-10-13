"""Sequence replay for a jointly trained teacher and recurrent visual student.

Current observations are stored once, in environment/time order.  Only episode
endpoints need an extra next observation: the following environment observation
can already belong to a reset episode (or be a reset placeholder).

Hidden checkpoints are incoming rollout states, not exact states of the current
perception model.  Replaying the burn-in with current weights mitigates, but does
not eliminate, that representation mismatch.

This module deliberately depends only on PyTorch and the standard library so its
replay semantics can be tested without a simulator or the FlashSAC dependency.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Optional, Sequence

import torch


def encode_depth(depth: torch.Tensor, num_bins: int = 100) -> torch.Tensor:
    """Store normalized, pre-quantized depth as its integer bin number.

    For VAIC's existing ``floor(depth * 100) / 100`` preprocessing this preserves
    all 101 possible values.  ``round`` is intentional: direct integer conversion
    can turn, for example, float32 0.53 into bin 52.  This is not /255 encoding.
    Already encoded uint8 tensors are accepted without changing their values.
    """
    if not 1 <= num_bins <= 255:
        raise ValueError("num_bins must be between 1 and 255 for uint8 storage")
    if depth.dtype == torch.uint8:
        return depth
    return depth.mul(num_bins).round().clamp(0, num_bins).to(torch.uint8)


def decode_depth(depth: torch.Tensor, num_bins: int = 100) -> torch.Tensor:
    if not 1 <= num_bins <= 255:
        raise ValueError("num_bins must be between 1 and 255")
    return depth.float().div(num_bins)


@dataclass
class ReplaySequenceBatch:
    """A balanced collection of chunks, with batch-major sequence tensors.

    ``indices`` selects the learning observations from ``obs``.  A nonterminal
    ``bootstrap_index`` selects the bootstrap observation.  A terminal index
    selects the *last action's current observation*: process ``terminal_obs``
    from the hidden AFTER that index, before any reset, to obtain its belief.
    ``bootstrap_discount`` already contains gamma to the actual horizon power.

    ``sample_weight`` sums to one, with one half assigned to each rollout source
    when both are present. It handles different counts of valid transitions in
    equally many teacher and student chunks. Use it for flattened scalar losses.
    """

    obs: torch.Tensor
    depth: torch.Tensor
    is_init: torch.Tensor
    valid: torch.Tensor
    h0: torch.Tensor
    checkpoint_step: torch.Tensor
    checkpoint_version: torch.Tensor
    source: torch.Tensor
    learning_start: torch.Tensor
    indices: torch.Tensor
    actions: torch.Tensor
    learning_mask: torch.Tensor
    nstep_reward: torch.Tensor
    bootstrap_discount: torch.Tensor
    bootstrap_index: torch.Tensor
    terminal_mask: torch.Tensor
    terminal_obs: torch.Tensor
    terminal_depth: torch.Tensor
    sample_weight: torch.Tensor

    def to(self, device, non_blocking: bool = False):
        # Transfer depth in its compact form and let the learner decode on GPU.
        float_fields = {"obs", "terminal_obs", "h0"}
        return type(self)(**{
            field.name: getattr(self, field.name).to(
                device=device,
                dtype=torch.float32 if field.name in float_fields else None,
                non_blocking=non_blocking,
            )
            for field in fields(self)
        })


@dataclass
class _TerminalRecord:
    env_ids: torch.Tensor
    obs: torch.Tensor
    depth: torch.Tensor


class OneStageSequenceReplay:
    """FIFO replay with sparse belief checkpoints and source-balanced chunks.

    ``capacity`` is a requested transition count, rounded down to complete rows
    of ``num_envs``. Environment indices ``[0, num_envs // 2)`` are the teacher
    source and the remaining indices are the student source.

    For interval=16, burn_in=16, sequence_length=32 and n_step=3, sampled chunks
    contain 66 observations. Their learning start is 16--31 observations after
    a checkpoint. The trailing observations supply n-step bootstrap states.
    Replay must hold at least 81 rows to allow this sampling after ring wrap.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        depth_shape: Sequence[int],
        hidden_dim: int = 256,
        checkpoint_interval: int = 16,
        burn_in: int = 16,
        sequence_length: int = 32,
        n_step: int = 3,
        gamma: float = 0.99,
        num_depth_bins: int = 100,
        obs_dtype: torch.dtype = torch.float16,
        storage_device="cpu",
        device=None,
        batch_size: int = 64,
        min_length: int = 0,
        check_memory: bool = True,
    ):
        if num_envs < 2 or num_envs % 2:
            raise ValueError("num_envs must be positive and even for 50:50 rollout sources")
        if checkpoint_interval < 1 or burn_in < 0 or sequence_length < 1 or n_step < 1:
            raise ValueError("Invalid checkpoint, burn-in, sequence, or n-step length")
        if hidden_dim < 1 or obs_dim < 1 or action_dim < 1:
            raise ValueError("Observation, action, and hidden dimensions must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not 1 <= num_depth_bins <= 255:
            raise ValueError("num_depth_bins must fit uint8")
        if obs_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("obs_dtype must be float16, bfloat16, or float32")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size is a positive even number of chunks")
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.depth_shape = tuple(int(x) for x in depth_shape)
        if not self.depth_shape or any(x < 1 for x in self.depth_shape):
            raise ValueError("depth_shape must contain positive dimensions")
        self.hidden_dim = int(hidden_dim)
        self.checkpoint_interval = int(checkpoint_interval)
        self.burn_in = int(burn_in)
        self.sequence_length = int(sequence_length)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.num_depth_bins = int(num_depth_bins)
        self.obs_dtype = obs_dtype
        self.storage = torch.device(storage_device)
        self.device = torch.device(device) if device is not None else self.storage
        self.batch_size = int(batch_size)
        self.min_length = int(min_length)
        self.total_sequence_length = (
            self.burn_in + self.checkpoint_interval - 1 + self.sequence_length + self.n_step
        )
        self.T = int(capacity) // self.num_envs
        minimum_rows = self.total_sequence_length + self.checkpoint_interval - 1
        if self.T < minimum_rows:
            raise ValueError(
                f"Replay capacity must hold at least {minimum_rows} steps per environment "
                f"({minimum_rows * self.num_envs} transitions) for the requested chunk layout"
            )
        self._checkpoint_rows = math.ceil(self.T / self.checkpoint_interval) + 1
        self.step = 0
        self._terminal_next = {}
        required = self.estimated_nbytes()
        if check_memory:
            self._check_memory(required)
        shape = (self.T, self.num_envs)
        self.obs = torch.empty(*shape, self.obs_dim, dtype=obs_dtype, device=self.storage)
        self.depth = torch.empty(*shape, *self.depth_shape, dtype=torch.uint8, device=self.storage)
        self.action = torch.empty(*shape, self.action_dim, device=self.storage)
        self.reward = torch.empty(shape, device=self.storage)
        self.discount = torch.empty(shape, device=self.storage)
        self.done = torch.empty(shape, dtype=torch.bool, device=self.storage)
        self.cut = torch.empty(shape, dtype=torch.bool, device=self.storage)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=self.storage)
        self.is_init = torch.empty(shape, dtype=torch.bool, device=self.storage)
        self.row_step = torch.full((self.T,), -1, dtype=torch.int64, device=self.storage)
        # Prefix sums at row boundaries make source-availability queries O(1).
        # T+1 boundaries are needed to retain the prefix immediately before the
        # oldest row as well as the boundary after the newest row on ring wrap.
        self._valid_prefix = torch.zeros((self.T + 1, 2), dtype=torch.int64, device=self.storage)
        self.checkpoint_hidden = torch.empty(
            self._checkpoint_rows, self.num_envs, self.hidden_dim,
            dtype=torch.float16, device=self.storage,
        )
        self.checkpoint_steps = torch.full(
            (self._checkpoint_rows,), -1, dtype=torch.int64, device=self.storage
        )
        self.checkpoint_versions = torch.zeros(
            (self._checkpoint_rows,), dtype=torch.int64, device=self.storage
        )

    @property
    def capacity(self) -> int:
        return self.T * self.num_envs

    def estimated_nbytes(self) -> int:
        """Fixed tensor allocation; sparse episode endpoints are additional."""
        obs_bytes = torch.empty((), dtype=self.obs_dtype).element_size()
        row_bytes = (
            self.obs_dim * obs_bytes + math.prod(self.depth_shape)
            + self.action_dim * 4 + 2 * 4 + 4
        )
        return (
            self.capacity * row_bytes + self.T * 8 + (self.T + 1) * 2 * 8
            + self._checkpoint_rows * (self.num_envs * self.hidden_dim * 2 + 16)
        )

    def nbytes(self) -> int:
        """Allocated tensor payload, including retained final observations."""
        dense = sum(
            value.numel() * value.element_size()
            for value in vars(self).values() if isinstance(value, torch.Tensor)
        )
        sparse = sum(
            tensor.numel() * tensor.element_size()
            for record in self._terminal_next.values()
            for tensor in (record.env_ids, record.obs, record.depth)
        )
        return dense + sparse

    def _check_memory(self, required: int):
        # Linux reports available RAM (including reclaimable caches), rather than
        # merely free RAM. In other environments, leave the check to allocation.
        if self.storage.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.storage)
            reserve = min(4 * 2**30, int(free * 0.25))
        elif self.storage.type == "cpu":
            try:
                entries = {
                    line.split(":", 1)[0]: int(line.split()[1]) * 1024
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                }
                free = entries["MemAvailable"]
            except (OSError, KeyError, ValueError):
                return
            reserve = max(2**30, int(free * 0.1))
        else:
            return
        if required > max(0, free - reserve):
            raise MemoryError(
                f"One-stage replay needs at least {required / 2**30:.2f} GiB on {self.storage}, "
                f"but only {free / 2**30:.2f} GiB is available before reserve. "
                "Reduce buffer capacity; terminal endpoints and training batches need extra memory."
            )

    def __len__(self) -> int:
        return min(self.step, self.T) * self.num_envs

    def _anchor_bounds(self):
        oldest = max(0, self.step - self.T)
        first = (oldest + self.checkpoint_interval - 1) // self.checkpoint_interval
        last = (self.step - self.total_sequence_length) // self.checkpoint_interval
        return first, last

    def can_sample(self, min_length: Optional[int] = None) -> bool:
        first, last = self._anchor_bounds()
        if len(self) < (self.min_length if min_length is None else min_length) or last < first:
            return False
        # Avoid accepting a buffer containing only reset placeholders. Sampling
        # performs a more precise per-chunk check, with a bounded rejection loop.
        start = first * self.checkpoint_interval + self.burn_in
        end = last * self.checkpoint_interval + self.total_sequence_length - self.n_step
        counts = self._valid_prefix[end % (self.T + 1)] - self._valid_prefix[start % (self.T + 1)]
        return bool((counts > 0).all())

    @torch.no_grad()
    def add(
        self, obs, depth, h_pre, is_init, action, reward, discount, done, cut,
        valid, next_obs, next_depth, version: int = 0,
    ):
        """Append one actual transition per environment (including invalid rows).

        ``h_pre`` is the belief *before* the current observation. ``is_init``
        resets that belief before processing the observation. ``valid=False``
        denotes reset placeholders, and excludes any n-step target traversing
        that row. ``discount`` is the environment's soft continuation factor;
        gamma is applied by this replay. ``done`` stops the return; ``cut`` also
        disables its bootstrap (true terminal vs. time-limit truncation).
        """
        expected = (self.num_envs, self.obs_dim)
        if tuple(obs.shape) != expected or tuple(next_obs.shape) != expected:
            raise ValueError(f"obs and next_obs must have shape {expected}")
        if tuple(depth.shape) != (self.num_envs, *self.depth_shape):
            raise ValueError("depth has the wrong shape")
        if tuple(next_depth.shape) != tuple(depth.shape):
            raise ValueError("next_depth has the wrong shape")
        if tuple(h_pre.shape) != (self.num_envs, self.hidden_dim):
            raise ValueError("h_pre must be [num_envs, hidden_dim]")
        if tuple(action.shape) != (self.num_envs, self.action_dim):
            raise ValueError("action has the wrong shape")
        g = self.step
        row = g % self.T
        # Drop the previous occupant's sparse endpoints together with the row.
        self._terminal_next.pop(g - self.T, None)
        self.obs[row].copy_(obs)
        self.depth[row].copy_(encode_depth(depth, self.num_depth_bins))
        self.action[row].copy_(action)
        for name, value in (
            ("reward", reward), ("discount", discount), ("done", done),
            ("cut", cut), ("valid", valid), ("is_init", is_init),
        ):
            getattr(self, name)[row].copy_(value.reshape(self.num_envs))
        self.row_step[row] = g
        half = self.num_envs // 2
        counts = self.valid[row].reshape(2, half).sum(dim=1)
        self._valid_prefix[(g + 1) % (self.T + 1)] = (
            self._valid_prefix[g % (self.T + 1)] + counts
        )
        if g % self.checkpoint_interval == 0:
            slot = (g // self.checkpoint_interval) % self._checkpoint_rows
            self.checkpoint_hidden[slot].copy_(h_pre)
            self.checkpoint_steps[slot] = g
            self.checkpoint_versions[slot] = int(version)
        # Gather on the source device before transferring, so normal next frames
        # do not consume either replay RAM or CPU/GPU transfer bandwidth.
        done_ids = done.reshape(-1).bool().nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self._terminal_next[g] = _TerminalRecord(
                done_ids.to(self.storage).clone(),
                next_obs.index_select(0, done_ids.to(next_obs.device)).to(
                    device=self.storage, dtype=self.obs_dtype, copy=True
                ),
                encode_depth(
                    next_depth.index_select(0, done_ids.to(next_depth.device)), self.num_depth_bins
                ).to(device=self.storage, copy=True),
            )
        self.step += 1

    def _learning_masks(self, checkpoints, envs, offsets):
        starts = self.burn_in + offsets
        indices = starts[:, None] + torch.arange(self.sequence_length, device=self.storage)
        absolute = checkpoints[:, None] + indices
        env_grid = envs[:, None].expand_as(indices)
        mask = self.valid[absolute % self.T, env_grid].clone()
        alive = torch.ones_like(mask)
        for k in range(self.n_step):
            at = (absolute + k) % self.T
            # A reset without a recorded done endpoint is not a valid temporal
            # link. Normal resets after done do not affect the stopped return.
            coherent = self.valid[at, env_grid]
            if k:
                coherent = coherent & ~self.is_init[at, env_grid]
            mask &= ~alive | coherent
            alive &= ~(self.done[at, env_grid] | self.cut[at, env_grid])
        nxt = (absolute + self.n_step) % self.T
        mask &= ~alive | (self.valid[nxt, env_grid] & ~self.is_init[nxt, env_grid])
        return starts, indices, mask

    @torch.no_grad()
    def sample(self, batch_size: Optional[int] = None) -> ReplaySequenceBatch:
        count = self.batch_size if batch_size is None else int(batch_size)
        if count < 2 or count % 2:
            raise ValueError("sample batch_size must be an even number of chunks")
        if not self.can_sample():
            raise RuntimeError("Replay has no complete balanced sequence batch yet")
        first, last = self._anchor_bounds()
        # Draw from each fixed source equally. Reject chunks with zero valid
        # learning transitions, but never spin forever on pathological data.
        half = self.num_envs // 2
        source = torch.arange(count, device=self.storage) // (count // 2)
        envs = torch.empty(count, dtype=torch.int64, device=self.storage)
        checkpoints = torch.empty_like(envs)
        offsets = torch.empty_like(envs)
        pending = torch.arange(count, device=self.storage)
        for _ in range(64):
            size = pending.numel()
            if not size:
                break
            envs[pending] = torch.randint(half, (size,), device=self.storage) + source[pending] * half
            checkpoints[pending] = torch.randint(first, last + 1, (size,), device=self.storage) * self.checkpoint_interval
            offsets[pending] = torch.randint(self.checkpoint_interval, (size,), device=self.storage)
            _, _, mask = self._learning_masks(checkpoints[pending], envs[pending], offsets[pending])
            pending = pending[~mask.any(dim=1)]
        if pending.numel():
            raise RuntimeError(
                "Could not sample valid teacher/student chunks after 64 attempts; "
                "collect more valid transitions or inspect reset/valid flags"
            )
        return self.gather(checkpoints, envs, offsets).to(self.device)

    @torch.no_grad()
    def gather(self, checkpoint_steps, env_ids, learning_offsets=None) -> ReplaySequenceBatch:
        """Deterministic chunk access using absolute (not ring) checkpoint steps."""
        checkpoints = torch.as_tensor(checkpoint_steps, dtype=torch.int64, device=self.storage).reshape(-1)
        envs = torch.as_tensor(env_ids, dtype=torch.int64, device=self.storage).reshape(-1)
        if learning_offsets is None:
            offsets = torch.zeros_like(checkpoints)
        else:
            offsets = torch.as_tensor(learning_offsets, dtype=torch.int64, device=self.storage).reshape(-1)
        if checkpoints.numel() == 0 or envs.shape != checkpoints.shape or offsets.shape != checkpoints.shape:
            raise ValueError("checkpoints, env_ids, and learning_offsets need equal nonzero lengths")
        if bool(((envs < 0) | (envs >= self.num_envs)).any()):
            raise ValueError("Invalid environment index")
        if bool(((offsets < 0) | (offsets >= self.checkpoint_interval)).any()):
            raise ValueError("learning_offsets must be less than checkpoint_interval")
        first, last = self._anchor_bounds()
        valid_anchors = (
            (checkpoints % self.checkpoint_interval == 0)
            & (checkpoints >= first * self.checkpoint_interval)
            & (checkpoints <= last * self.checkpoint_interval)
        )
        if not bool(valid_anchors.all()):
            raise ValueError("Checkpoint is overwritten, unaligned, or has incomplete future observations")
        checkpoint_slots = (checkpoints // self.checkpoint_interval) % self._checkpoint_rows
        if not bool((self.checkpoint_steps[checkpoint_slots] == checkpoints).all()):
            raise RuntimeError("Hidden checkpoint no longer matches the requested observations")
        absolute = checkpoints[:, None] + torch.arange(self.total_sequence_length, device=self.storage)
        rows = absolute % self.T
        if not bool((self.row_step[rows] == absolute).all()):
            raise RuntimeError("Sequence contains an overwritten or missing replay row")
        env_grid = envs[:, None].expand_as(rows)
        starts, indices, mask = self._learning_masks(checkpoints, envs, offsets)
        learning_abs = checkpoints[:, None] + indices
        learning_env = envs[:, None].expand_as(indices)
        reward = torch.zeros_like(indices, dtype=torch.float32)
        scale = torch.ones_like(reward)
        alive = torch.ones_like(mask)
        bootstrap_discount = torch.zeros_like(reward)
        bootstrap_index = indices + self.n_step
        terminal_mask = torch.zeros_like(mask)
        terminal_step = torch.full_like(indices, -1)
        for k in range(self.n_step):
            step_rows = (learning_abs + k) % self.T
            reward += alive * (self.gamma**k) * scale * self.reward[step_rows, learning_env]
            scale = torch.where(alive, scale * self.discount[step_rows, learning_env], scale)
            done = self.done[step_rows, learning_env]
            cut = self.cut[step_rows, learning_env]
            ends = alive & (done | cut)
            bootstrap_index = torch.where(ends, indices + k, bootstrap_index)
            terminal_mask |= ends & done
            terminal_step = torch.where(ends & done, learning_abs + k, terminal_step)
            bootstrap_discount = torch.where(
                ends, (self.gamma ** (k + 1)) * scale * ~cut, bootstrap_discount
            )
            alive &= ~ends
        bootstrap_discount = torch.where(alive, (self.gamma**self.n_step) * scale, bootstrap_discount)
        # Invalid targets are fully neutral, including their endpoint branch.
        reward = torch.where(mask, reward, 0.0)
        bootstrap_discount = torch.where(mask, bootstrap_discount, 0.0)
        terminal_mask &= mask
        shape = (*indices.shape,)
        terminal_obs = torch.zeros(*shape, self.obs_dim, dtype=self.obs_dtype, device=self.storage)
        terminal_depth = torch.zeros(*shape, *self.depth_shape, dtype=torch.uint8, device=self.storage)
        for absolute_step in terminal_step[terminal_mask].unique().tolist():
            record = self._terminal_next.get(absolute_step)
            if record is None:
                raise RuntimeError("Missing terminal next observation for a retained replay row")
            b, t = (terminal_mask & (terminal_step == absolute_step)).nonzero(as_tuple=True)
            requested_envs = envs[b]
            positions = torch.searchsorted(record.env_ids, requested_envs)
            if bool((positions >= record.env_ids.numel()).any()) or not bool(
                (record.env_ids[positions] == requested_envs).all()
            ):
                raise RuntimeError("Terminal observation does not match the requested environment")
            terminal_obs[b, t] = record.obs[positions]
            terminal_depth[b, t] = record.depth[positions]
        source = (envs >= self.num_envs // 2).long()
        weights = torch.zeros_like(reward)
        present = [s for s in (0, 1) if bool((mask & (source[:, None] == s)).any())]
        for s in present:
            selected = mask & (source[:, None] == s)
            weights[selected] = 1.0 / (len(present) * selected.sum())
        return ReplaySequenceBatch(
            obs=self.obs[rows, env_grid], depth=self.depth[rows, env_grid],
            is_init=self.is_init[rows, env_grid], valid=self.valid[rows, env_grid],
            h0=self.checkpoint_hidden[checkpoint_slots, envs],
            checkpoint_step=checkpoints, checkpoint_version=self.checkpoint_versions[checkpoint_slots],
            source=source, learning_start=starts, indices=indices,
            actions=self.action[learning_abs % self.T, learning_env], learning_mask=mask,
            nstep_reward=reward.unsqueeze(-1), bootstrap_discount=bootstrap_discount.unsqueeze(-1),
            bootstrap_index=bootstrap_index, terminal_mask=terminal_mask,
            terminal_obs=terminal_obs, terminal_depth=terminal_depth, sample_weight=weights,
        )
