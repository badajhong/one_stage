"""Replay invariants that do not require Isaac Sim or a running environment."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "active_adaptation/learning/flashsac/one_stage_replay.py"
)
SPEC = importlib.util.spec_from_file_location("one_stage_replay_under_test", MODULE_PATH)
replay_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay_module
SPEC.loader.exec_module(replay_module)
OneStageSequenceReplay = replay_module.OneStageSequenceReplay
encode_depth = replay_module.encode_depth
decode_depth = replay_module.decode_depth


def make_replay(**kwargs):
    options = dict(
        num_envs=4, capacity=80, obs_dim=2, action_dim=1, depth_shape=(1, 2, 2),
        hidden_dim=3, checkpoint_interval=2, burn_in=2, sequence_length=3,
        n_step=3, gamma=0.9, batch_size=4, check_memory=False,
    )
    options.update(kwargs)
    return OneStageSequenceReplay(**options)


def add_step(replay, t, *, dones=(), cuts=(), inits=(), invalid=(), soft_discount=1.0):
    n = replay.num_envs
    env = torch.arange(n)
    obs = torch.stack((torch.full((n,), float(t)), env.float()), -1)
    next_obs = obs.clone()
    next_obs[:, 0] += 1
    depth = torch.full((n, *replay.depth_shape), (t % 101) / 100)
    next_depth = torch.full_like(depth, ((t + 1) % 101) / 100)
    done = torch.isin(env, torch.tensor(dones))
    cut = torch.isin(env, torch.tensor(cuts))
    if done.any():
        # Final frames are deliberately different from next iteration's reset.
        next_obs[done, 0] = 1000 + t
        next_depth[done] = 0.99
    replay.add(
        obs=obs, depth=depth, h_pre=torch.full((n, replay.hidden_dim), float(t)),
        is_init=torch.isin(env, torch.tensor(inits)), action=torch.full((n, 1), t / 100),
        reward=torch.full((n,), float(t + 1)), discount=torch.full((n,), soft_discount),
        done=done, cut=cut, valid=~torch.isin(env, torch.tensor(invalid)),
        next_obs=next_obs, next_depth=next_depth, version=t // 2,
    )


def test_uint8_preserves_all_existing_depth_bins_and_uses_half_fp16_space():
    depth = torch.arange(101, dtype=torch.float32) / 100
    encoded = encode_depth(depth)
    assert encoded.dtype == torch.uint8
    assert torch.equal(encoded, torch.arange(101, dtype=torch.uint8))
    assert torch.equal(decode_depth(encoded), depth)
    assert encoded.nbytes * 2 == depth.half().nbytes
    # Direct truncation is incorrect for at least these float32 values.
    assert torch.equal(encode_depth(torch.tensor([0.53, 0.59])), torch.tensor([53, 59], dtype=torch.uint8))
    assert not torch.equal(depth.half().float(), depth)


def test_nstep_shortens_at_truncation_and_uses_true_terminal_observation():
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, dones=(0, 2) if t == 4 else (),
                 inits=(0, 2) if t == 5 else (), soft_discount=0.8)
    batch = replay.gather([0, 0], [0, 2])
    # Learning t=2,3,4 ends at t=4 rather than reading the reset observation t=5.
    factor = 0.9 * 0.8
    expected_reward = torch.tensor([3 + factor * 4 + factor**2 * 5, 4 + factor * 5, 5])
    expected_discount = torch.tensor([factor**3, factor**2, factor])
    torch.testing.assert_close(batch.nstep_reward[0, :, 0], expected_reward)
    torch.testing.assert_close(batch.bootstrap_discount[0, :, 0], expected_discount)
    assert batch.terminal_mask.all()
    assert torch.equal(batch.bootstrap_index, torch.full((2, 3), 4))
    assert torch.equal(batch.terminal_obs[:, :, 0], torch.full((2, 3), 1004.0, dtype=torch.float16))
    assert (batch.terminal_depth == 99).all()
    assert (batch.obs[:, 5, 0] == 5).all()
    assert batch.is_init[:, 5].all()
    assert batch.learning_mask.all()


def test_true_terminal_zeroes_bootstrap_and_never_includes_new_episode_rewards():
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, dones=(1,) if t == 3 else (), cuts=(1,) if t == 3 else (),
                 inits=(1,) if t == 4 else ())
    batch = replay.gather([0], [1])
    torch.testing.assert_close(batch.nstep_reward[0, :2, 0], torch.tensor([3 + 0.9 * 4, 4]))
    assert (batch.bootstrap_discount[0, :2] == 0).all()
    assert batch.terminal_mask[0, :2].all()
    # The following reset row can begin a new valid trajectory, provided the
    # environment marked it valid; its belief is reset separately by is_init.
    assert batch.learning_mask[0, 2]
    assert not batch.terminal_mask[0, 2]
    assert batch.bootstrap_index[0, 2] == 7


def test_continuing_target_uses_gamma_horizon_and_soft_discounts_once():
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, soft_discount=0.5)
    batch = replay.gather([0], [0])
    factor = 0.9 * 0.5
    torch.testing.assert_close(
        batch.nstep_reward[0, :, 0],
        torch.tensor([3 + factor * 4 + factor**2 * 5,
                      4 + factor * 5 + factor**2 * 6,
                      5 + factor * 6 + factor**2 * 7]),
    )
    torch.testing.assert_close(batch.bootstrap_discount, torch.full((1, 3, 1), factor**3))
    assert torch.equal(batch.bootstrap_index, torch.tensor([[5, 6, 7]]))
    assert not batch.terminal_mask.any()


@pytest.mark.parametrize("reset_only", [False, True])
def test_invalid_or_unrecorded_reset_link_does_not_cross_into_nstep_target(reset_only):
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, invalid=(0,) if t == 4 and not reset_only else (),
                 inits=(0,) if t == 4 and reset_only else ())
    batch = replay.gather([0], [0])
    assert not batch.learning_mask[0, :2].any()
    assert bool(batch.learning_mask[0, 2]) == reset_only
    assert (batch.nstep_reward[0, :2] == 0).all()
    assert (batch.bootstrap_discount[0, :2] == 0).all()


def test_nstep_does_not_bootstrap_from_a_reset_placeholder_without_done():
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, invalid=(0,) if t == 5 else ())
    batch = replay.gather([0], [0])
    assert not batch.learning_mask.any()
    assert not batch.sample_weight.any()


def test_reset_placeholder_after_recorded_done_does_not_invalidate_terminal_target():
    replay = make_replay()
    for t in range(12):
        add_step(replay, t, dones=(0,) if t == 4 else (),
                 invalid=(0,) if t == 5 else (), inits=(0,) if t == 5 else ())
    batch = replay.gather([0], [0])
    assert batch.learning_mask.all()
    assert batch.terminal_mask.all()


def test_ring_wrap_preserves_absolute_order_and_drops_overwritten_endpoints():
    replay = make_replay(capacity=48)
    for t in range(40):
        add_step(replay, t, dones=(0,) if t in (5, 35) else (),
                 inits=(0,) if t in (6, 36) else ())
    assert len(replay) == replay.capacity == 48
    assert set(replay._terminal_next) == {35}
    batch = replay.gather([28, 30], [0, 2], [1, 0])
    assert torch.equal(batch.obs[0, :, 0], torch.arange(28, 37).half())
    assert torch.equal(batch.obs[1, :, 0], torch.arange(30, 39).half())
    assert torch.equal(batch.h0[:, 0], torch.tensor([28, 30], dtype=torch.float16))
    assert torch.equal(batch.checkpoint_version, torch.tensor([14, 15]))
    assert torch.equal(batch.learning_start, torch.tensor([3, 2]))
    with pytest.raises(ValueError, match="overwritten"):
        replay.gather([0], [0])
    with pytest.raises(ValueError, match="unaligned"):
        replay.gather([29], [0])
    for _ in range(10):
        sampled = replay.sample(4)
        assert ((sampled.checkpoint_step >= 28) & (sampled.checkpoint_step <= 30)).all()


def test_balanced_sampling_weights_correct_different_valid_counts_and_preserves_dtype():
    torch.manual_seed(9)
    replay = make_replay()
    for t in range(20):
        # Different source valid fractions, while each source retains usable chunks.
        add_step(replay, t, invalid=(0, 1) if t in (5, 13) else ())
    batch = replay.sample(20)
    assert torch.equal(torch.bincount(batch.source), torch.tensor([10, 10]))
    for source in (0, 1):
        torch.testing.assert_close(batch.sample_weight[batch.source == source].sum(), torch.tensor(0.5))
    assert (batch.sample_weight[~batch.learning_mask] == 0).all()
    assert batch.depth.dtype == batch.terminal_depth.dtype == torch.uint8
    assert batch.obs.dtype == batch.terminal_obs.dtype == batch.h0.dtype == torch.float32
    assert replay.obs.dtype == torch.float16
    assert replay.depth.dtype == torch.uint8


def test_sparse_checkpoint_memory_and_capacity_estimate():
    replay = make_replay(capacity=4 * 120, checkpoint_interval=16, burn_in=16, sequence_length=32)
    assert replay.nbytes() == replay.estimated_nbytes()
    assert replay.checkpoint_hidden.shape == (9, 4, 3)
    base_bytes = replay.nbytes()
    add_step(replay, 0, dones=(0, 2))
    assert replay.nbytes() == base_bytes + 2 * (8 + 2 * 2 + 4)
    assert not replay.can_sample()
    with pytest.raises(ValueError, match="81 steps"):
        make_replay(capacity=4 * 80, checkpoint_interval=16, burn_in=16, sequence_length=32)


def test_no_valid_student_source_is_not_sampleable():
    replay = make_replay()
    for t in range(20):
        add_step(replay, t, invalid=(2, 3))
    assert not replay.can_sample()
    with pytest.raises(RuntimeError, match="balanced"):
        replay.sample()


def test_default_chunk_layout_supports_16_to_31_step_burnin():
    replay = make_replay(capacity=4 * 100, checkpoint_interval=16, burn_in=16, sequence_length=32)
    for t in range(90):
        add_step(replay, t)
    batch = replay.gather([0, 16], [0, 2], [0, 15])
    assert batch.obs.shape == (2, 66, 2)
    assert torch.equal(batch.learning_start, torch.tensor([16, 31]))
    assert torch.equal(batch.indices[:, -1], torch.tensor([47, 62]))
    assert torch.equal(batch.bootstrap_index[:, -1], torch.tensor([50, 65]))
    assert batch.learning_mask.all()


def test_memory_guard_precedes_tensor_allocation(monkeypatch):
    checked = []

    def deny(self, required):
        checked.append(required)
        assert not hasattr(self, "obs")
        raise MemoryError("test capacity guard")

    monkeypatch.setattr(OneStageSequenceReplay, "_check_memory", deny)
    with pytest.raises(MemoryError, match="test capacity guard"):
        make_replay(check_memory=True)
    assert checked[0] > 0


def test_constant_time_availability_matches_reference_after_repeated_ring_wraps():
    # A non-divisible ring length exercises both independent ring moduli:
    # observation rows have length T, cumulative-count boundaries have T+1.
    replay = make_replay(capacity=4 * 13)
    for t in range(100):
        if t < 16 or 61 <= t < 80:
            invalid = (2, 3)
        elif 26 <= t < 61:
            invalid = (0, 1)
        else:
            invalid = ()
        add_step(replay, t, invalid=invalid)
        first, last = replay._anchor_bounds()
        if last < first:
            expected = False
        else:
            start = first * replay.checkpoint_interval + replay.burn_in
            end = last * replay.checkpoint_interval + replay.total_sequence_length - replay.n_step
            rows = torch.arange(start, end) % replay.T
            expected = bool(replay.valid[rows, :2].any() and replay.valid[rows, 2:].any())
        assert replay.can_sample() == expected
    assert replay.can_sample()
    # can_sample must query compact cumulative metadata, never scan all stored
    # validity flags. Enforce this without a timing-sensitive performance test.
    class NoFullReplayRead:
        def __getitem__(self, _):
            raise AssertionError("can_sample scanned replay validity storage")

    replay.valid = NoFullReplayRead()
    assert replay.can_sample()


def test_sparse_hidden_ring_never_reuses_an_overwritten_checkpoint():
    replay = make_replay(capacity=4 * 13)
    for t in range(150):
        add_step(replay, t)
        first, last = replay._anchor_bounds()
        if last < first:
            continue
        anchors = torch.arange(first, last + 1) * replay.checkpoint_interval
        envs = torch.zeros_like(anchors)
        batch = replay.gather(anchors, envs)
        assert torch.equal(batch.h0[:, 0], anchors.half())
        assert torch.equal(batch.checkpoint_version, anchors // 2)
    assert replay.nbytes() == replay.estimated_nbytes()
