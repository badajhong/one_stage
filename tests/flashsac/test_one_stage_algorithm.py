"""End-to-end CPU joint learner tests with real networks and a small fake env."""

import copy
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import pytest
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.flashsac.one_stage_flashsac_vel import (
    OneStageFlashSACVel, OneStageFlashSACVelConfig,
)
from active_adaptation.learning.flashsac.one_stage_update import _teacher_pull_loss


torch.set_num_threads(2)


class FakeEnv:
    num_envs = 4
    max_episode_length = 9
    device = torch.device("cpu")
    current_iter = 7

    def __init__(self):
        self.transform = SimpleNamespace(transforms=[])
        self.action_manager = SimpleNamespace(joint_names=["j0", "j1", "j2"])
        self.geometry = torch.linspace(-1, 1, 384).reshape(128, 3)
        self.shapes = {
            "command": (10,), "policy": (12,), "priv": (14,), "object_": (21,),
            "object_geo_": (384,), "ref_joint_pos_": (3,), "vel_command": (3,),
            "depth": (1, 32, 64),
        }
        self.observation_spec = Composite({
            key: Unbounded((4, *shape), device="cpu") for key, shape in self.shapes.items()
        }, shape=(4,), device="cpu")
        self.action_spec = Unbounded((4, 3), device="cpu")
        self.reward_spec = Unbounded((4, 2), device="cpu")
        self.observation_funcs = {}
        for key, shape in self.shapes.items():
            value = (self.geometry.flatten().expand(4, -1).clone() if key == "object_geo_"
                     else torch.zeros(4, *shape))
            function = lambda value=value: value.clone()
            self.observation_funcs[key] = SimpleNamespace(funcs={"all": function}, _compute=function)

    def set_progress(self, iteration):
        self.current_iter = iteration

    def observation(self, time=0):
        td = TensorDict({key: torch.randn(4, *shape) * 0.1
                         for key, shape in self.shapes.items()}, batch_size=[4])
        td["object_geo_"] = self.geometry.flatten().expand(4, -1).clone()
        # A valid object orientation, in the teacher's last-12 pose convention.
        td["object_"][:, -9:] = torch.eye(3).flatten()
        td["command"][:, 0] = time * 4 + torch.arange(4)
        td["depth"] = torch.randint(101, (4, 1, 32, 64)).float() / 100
        td["is_init"] = torch.zeros(4, 1, dtype=torch.bool)
        td["step_count"] = torch.full((4, 1), 3, dtype=torch.long)
        return td


def make_policy(**options):
    torch.manual_seed(51)
    env = FakeEnv()
    settings = dict(
        num_envs=4, num_env_steps=4096, train_every=4, sample_batch_size=16,
        sequence_length=4, perception_window_size=3, n_step=2,
        buffer_min_length=0, buffer_max_length=256, updates_per_interaction_step=.25,
        actor_num_blocks=1, actor_hidden_dim=16, critic_num_blocks=1, critic_hidden_dim=16,
        critic_state_encoder_dim=8, critic_action_encoder_dim=8, critic_num_bins=9,
        latent_dim=16, perception_model_dim=16, perception_ffn_dim=32, use_amp=False, use_compile=False,
        obs_drop_terms=(), obs_future_steps=None,
    )
    settings.update(options)
    cfg = OneStageFlashSACVelConfig(**settings)
    return OneStageFlashSACVel(cfg, env.observation_spec, env.action_spec, env.reward_spec, "cpu", env)


def test_default_window_model_has_no_recurrent_state_or_replay_options():
    policy = make_policy()
    assert policy.cfg.adapt_module == "window_transformer"
    assert not hasattr(policy.cfg, "burn_in") and not hasattr(policy.cfg, "belief_hidden_dim")
    assert not any(isinstance(m, (torch.nn.GRU, torch.nn.GRUCell)) for m in policy.perception.modules())
    assert policy.make_tensordict_primer() is None
    defaults = OneStageFlashSACVelConfig()
    assert (defaults.perception_window_size, defaults.perception_model_dim,
            defaults.perception_num_layers, defaults.perception_num_heads) == (16, 128, 1, 4)


def collect(policy, steps=22):
    """Populate real replay, retaining true final observations before reset."""
    rollout = policy.get_rollout_policy("train")
    carry = policy.env.observation()
    carry["is_init"].fill_(True)
    carry["step_count"].zero_()
    with torch.no_grad():
        for time in range(steps):
            td = rollout(carry)
            nxt = policy.env.observation(time + 1)
            nxt["step_count"] = td["step_count"] + 1
            terminated = torch.zeros(4, 1, dtype=torch.bool)
            truncated = torch.zeros_like(terminated)
            if time % 11 == 8:
                truncated[0] = True
                terminated[2] = True
            if time % 9 == 6:
                truncated[3] = True
            done = terminated | truncated
            nxt.update({
                "reward": torch.full((4, 2), .25), "discount": torch.full((4, 1), .8),
                "terminated": terminated, "truncated": truncated, "done": done,
            })
            td["next"] = nxt
            policy.add_transition(td)
            carry = nxt.clone()
            carry["is_init"] = done.clone()
            carry["step_count"][done] = 0
            # The reset state intentionally differs from the terminal state.
            carry["command"][done[:, 0], 0] = -100 - time
    assert policy.can_start_training()


def parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(before, module):
    return any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_train_rollout_assigns_half_to_each_actor_and_eval_needs_no_privileged_inputs():
    policy = make_policy()
    td = policy.env.observation()

    def sample(actor, observation, role, is_init):
        assert observation.shape[0] == 2
        return torch.full((2, 3), -.25 if role == "teacher" else .5)

    with patch.object(policy, "_sample_actor", side_effect=sample):
        result = policy.get_rollout_policy("train")(td.clone())
    torch.testing.assert_close(result["flashsac_u"][:2], torch.full((2, 3), -.25))
    torch.testing.assert_close(result["flashsac_u"][2:], torch.full((2, 3), .5))
    torch.testing.assert_close(result["action"], td["ref_joint_pos_"] + policy.cfg.residual_scale * result["flashsac_u"])
    assert "belief_hx" not in result
    assert not result["priv_pred"][:2].any()
    student_only = td.select("policy", "vel_command", "depth", "is_init", "step_count", "ref_joint_pos_")
    with patch.object(policy._actor, "apply", side_effect=AssertionError("student eval used teacher")):
        full_eval = policy.get_rollout_policy("eval")(td.clone())
        sparse_eval = policy.get_rollout_policy("student")(student_only.clone())
    torch.testing.assert_close(full_eval["action"], sparse_eval["action"], rtol=0, atol=0)


def test_replay_to_supervision_to_shared_sac_updates_and_detaches_perception():
    policy = make_policy()
    collect(policy)
    assert not policy.buffer.obs.is_inference()
    sequence = policy.buffer.sample()
    before_perception = parameters(policy.perception)
    perception_info = policy._train_perception(sequence)
    assert changed(before_perception, policy.perception)
    assert torch.isfinite(perception_info["perception/loss"])
    before_teacher = parameters(policy.actor)
    before_encoder = parameters(policy.actor.encoder_priv)
    before_student = parameters(policy.student_actor)
    before_critic = parameters(policy.critic)
    before_perception = parameters(policy.perception)
    info = policy._update_once()
    assert changed(before_teacher, policy.actor)
    assert changed(before_encoder, policy.actor.encoder_priv)
    assert changed(before_student, policy.student_actor)
    assert changed(before_critic, policy.critic)
    assert not changed(before_perception, policy.perception)
    assert all(parameter.grad is None for parameter in policy.perception.parameters())
    assert all(torch.isfinite(value).all() for value in info.values())


def test_public_update_starts_supervision_at_warmup_and_honors_fractional_sac_rate():
    policy = make_policy()
    policy.update()
    assert policy._perception_updates == policy._update_step == 0
    collect(policy)
    policy.update()
    assert policy._perception_updates == 1
    assert policy._update_step == 0 and policy._update_counter == .25
    for _ in range(3):
        collect(policy, steps=1)
        policy.update()
    assert policy._perception_updates == 2  # first warmup update, then step 24 boundary
    assert policy._update_step == 1 and policy._update_counter == 0
    info = policy.pop_info()
    assert "perception/loss" in info and "critic/loss" in info
    assert "sac/teacher/actor/loss" in info and "sac/student/actor/loss" in info
    assert "sac/student/actor/lr" in info
    assert not any(key.startswith(("train/", "teacher/", "student/")) for key in info)
    assert not policy._info_sum and not policy._info_cnt


def test_pull_only_trains_teacher_encoder_and_caps_gradient_norm():
    policy = make_policy()
    collect(policy)
    batch = policy._sac_batch(policy.buffer.sample())
    batch["actor_observation"].requires_grad_(True)
    batch["student_latent"] = (batch["student_latent"] + 1).detach().requires_grad_(True)
    teacher_z = policy.actor.encode_priv(batch["actor_observation"])
    rl_loss = teacher_z.square().mean()
    loss, info = _teacher_pull_loss(policy.actor, batch, rl_loss, 1000., .1)
    loss.backward()
    assert batch["student_latent"].grad is None
    assert batch["actor_observation"].grad is None
    assert all(p.grad is None for p in policy.perception.parameters())
    assert all(p.grad is None for p in policy.actor.policy.parameters())
    assert all(p.grad is None for p in policy.critic.parameters())
    norm = torch.stack([p.grad.square().sum() for p in policy.actor.encoder_priv.parameters()]).sum().sqrt()
    assert norm > 0
    assert info["pull_effective_coef"] < 1000.
    torch.testing.assert_close(norm, .1 * info["rl_encoder_grad_norm"], rtol=1e-4, atol=1e-8)


def test_pull_has_no_influence_when_rl_encoder_gradient_is_zero():
    policy = make_policy()
    collect(policy)
    batch = policy._sac_batch(policy.buffer.sample())
    rl_loss = policy.actor.encode_priv(batch["actor_observation"]).sum() * 0
    loss, info = _teacher_pull_loss(policy.actor, batch, rl_loss, .001, .1)
    assert loss.item() == info["pull_effective_coef"].item() == 0


def test_active_pull_shares_teacher_optimizer_step_and_keeps_perception_detached():
    policy = make_policy(teacher_pull_warmup_updates=0, teacher_pull_ramp_updates=0)
    collect(policy)
    before = parameters(policy.perception)
    with patch.object(policy._actor.optimizer, "step", wraps=policy._actor.optimizer.step) as step:
        info = policy._update_once()
    assert step.call_count == 1
    assert not changed(before, policy.perception)
    assert all(p.grad is None for p in policy.perception.parameters())
    assert info["teacher/actor/pull_effective_coef"] > 0
    assert info["teacher/actor/pull_grad_ratio"] <= .10001
    assert not any("pull" in key for key in info if key.startswith("student/"))
    assert all(torch.isfinite(value).all() for value in info.values())
    # Delayed SAC actor steps must not apply the auxiliary loss either.
    before_encoder = parameters(policy.actor.encoder_priv)
    info = policy._update_once()
    assert not changed(before_encoder, policy.actor.encoder_priv)
    assert "teacher/actor/pull_loss" not in info


def test_pull_schedule_defaults_disable_and_checkpoint_resume():
    policy = make_policy()
    assert policy.cfg.teacher_pull_enabled
    for count, expected in [(0, 0), (99, 0), (100, 0), (600, .0005), (1100, .001), (2000, .001)]:
        policy._perception_updates = count
        assert policy._teacher_pull_weight() == pytest.approx(expected)
    restored = make_policy()
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    assert restored._teacher_pull_weight() == policy._teacher_pull_weight()
    policy.cfg.teacher_pull_enabled = False
    assert policy._teacher_pull_weight() == 0
    policy.cfg.teacher_pull_enabled = True
    policy.cfg.teacher_pull_coef = 0
    assert policy._teacher_pull_weight() == 0


@pytest.mark.parametrize("key,value", [
    ("teacher_pull_coef", -1), ("teacher_pull_coef", float("nan")),
    ("teacher_pull_max_grad_ratio", -1), ("teacher_pull_max_grad_ratio", float("inf")),
    ("teacher_pull_warmup_updates", -1), ("teacher_pull_ramp_updates", 1.5),
])
def test_invalid_pull_settings_fail_early(key, value):
    with pytest.raises(ValueError, match=key):
        make_policy(**{key: value})


def test_supervision_recomputes_the_current_teacher_latent():
    policy = make_policy()
    collect(policy)
    sequence = policy.buffer.sample()
    with torch.no_grad():
        end = int(sequence.indices.max()) + 1
        predictions = policy.perception.sequence(
            sequence.depth[:, :end].float() / 100, policy._context(sequence.obs[:, :end]),
            sequence.is_init[:, :end], sequence.valid[:, :end],
        )[0]
        raw = policy._gather(sequence.obs, sequence.indices)[..., :policy.replay_obs_dim]
        old_target = policy.actor.encode_priv(raw.flatten(0, 1)).reshape_as(predictions)
        policy.actor.encoder_priv[-1].bias.add_(2.0)
        current_target = policy.actor.encode_priv(raw.flatten(0, 1)).reshape_as(predictions)
        expected_loss = ((predictions - current_target).square().mean(-1) * sequence.sample_weight).sum()
    assert not torch.allclose(old_target, current_target)
    info = policy._train_perception(sequence)
    torch.testing.assert_close(info["perception/priv_loss"], expected_loss)


def test_terminal_bootstrap_uses_true_endpoint_and_pre_reset_raw_history():
    policy = make_policy()
    collect(policy)
    sequence = policy.buffer.gather([4, 4], [0, 2]).to("cpu")
    terminal = sequence.terminal_mask & sequence.learning_mask
    assert terminal.any()
    rows, times = terminal.nonzero(as_tuple=True)
    ends = sequence.bootstrap_index[rows, times]
    prefix = policy.cfg.perception_window_size - 1
    indices = ends[:, None] + torch.arange(1-prefix, 1)
    final_depth = torch.cat([sequence.depth[rows[:, None], indices],
                             sequence.terminal_depth[terminal][:, None]], 1).float()/100
    final_obs = torch.cat([sequence.obs[rows[:, None], indices],
                           sequence.terminal_obs[terminal][:, None]], 1)
    flags = torch.zeros(rows.numel(), 1, dtype=torch.bool)
    inits = torch.cat([sequence.is_init[rows[:, None], indices], flags], 1)
    valid = torch.cat([sequence.valid[rows[:, None], indices], ~flags], 1)
    with torch.no_grad():
        expected = policy.perception(final_depth, policy._context(final_obs), inits, valid)[0]
    predictions = []
    original = policy.perception.window
    def capture(*args):
        value = original(*args)
        predictions.append(value[0].clone())
        return value
    with patch.object(policy.perception, "window", side_effect=capture):
        batch = policy._sac_batch(sequence)
    torch.testing.assert_close(predictions[-1], expected)
    environment_ids = policy.actor.field(batch["actor_observation"], "command")[:, 0].long() % 4
    assert (environment_ids < 2).sum() == policy.cfg.sample_batch_size // 2
    assert (environment_ids >= 2).sum() == policy.cfg.sample_batch_size // 2


def test_full_checkpoint_roundtrip_restores_models_optimizers_and_counters_with_empty_replay():
    policy = make_policy()
    collect(policy)
    policy._train_perception(policy.buffer.sample())
    policy._update_once()
    policy._update_counter = .5
    serialized = io.BytesIO()
    torch.save(policy.state_dict(), serialized)
    serialized.seek(0)
    saved = torch.load(serialized, weights_only=False)
    restored = make_policy()
    restored._info_sum["old/loss"] = torch.tensor(10.)
    restored._info_cnt["old/loss"] = 2
    restored.load_state_dict(saved)
    expected = copy.deepcopy(saved)
    # Episode accumulators restart alongside the empty replay; learned reward
    # statistics, network/optimizer states and training counters must survive.
    expected["reward_normalizer"]["G_r"] = torch.zeros_like(restored.reward_normalizer.G_r)
    assert_nested_equal(expected, restored.state_dict())
    assert restored.buffer is None and not restored._noise
    assert not restored._info_sum and not restored._info_cnt
    assert not restored.can_start_training()
    collect(restored)
    before = restored._update_step
    restored._update_once()
    assert restored._update_step == before + 1


@pytest.mark.parametrize("version", [None, 1])
def test_old_checkpoints_are_rejected_before_loading_any_weights(version):
    policy = make_policy()
    before = copy.deepcopy(policy.state_dict())
    old = copy.deepcopy(before)
    if version is None:
        old.pop("one_stage_version")
    else:
        old["one_stage_version"] = version
    with pytest.raises(ValueError, match="window Transformer checkpoints"):
        policy.load_state_dict(old)
    assert_nested_equal(before, policy.state_dict())


def test_teacher_rollout_skips_perception_and_train_only_caches_student_half():
    policy = make_policy()
    teacher = policy.get_rollout_policy("teacher")
    with patch.object(policy.perception, "encode", side_effect=AssertionError("teacher encoded depth")):
        teacher(policy.env.observation())
    train = policy.get_rollout_policy("train")
    train(policy.env.observation())
    assert train._history.valid.shape == (2, policy.cfg.perception_window_size)


@pytest.mark.parametrize("key,value", [
    ("perception_window_size", 0), ("perception_num_layers", 0),
    ("perception_model_dim", 15), ("perception_num_heads", 0),
    ("perception_ffn_dim", 0), ("adapt_module", "gru"),
])
def test_invalid_window_settings_fail_early(key, value):
    with pytest.raises(ValueError):
        make_policy(**{key: value})
