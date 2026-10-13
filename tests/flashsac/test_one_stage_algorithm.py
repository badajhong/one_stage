"""End-to-end CPU joint learner tests with real networks and a small fake env."""

import copy
from dataclasses import fields
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import pytest
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.flashsac.flashsac_vel import FlashSACVel, FlashSACVelConfig
from active_adaptation.learning.flashsac.one_stage_flashsac_vel import (
    BELIEF_KEY, OneStageFlashSACVel, OneStageFlashSACVelConfig,
)


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
        td[BELIEF_KEY] = torch.zeros(4, 16)
        return td


def make_policy(**options):
    torch.manual_seed(51)
    env = FakeEnv()
    settings = dict(
        num_envs=4, num_env_steps=4096, train_every=4, sample_batch_size=16,
        sequence_length=4, checkpoint_interval=2, burn_in=2, n_step=2,
        buffer_min_length=0, buffer_max_length=256, updates_per_interaction_step=.25,
        actor_num_blocks=1, actor_hidden_dim=16, critic_num_blocks=1, critic_hidden_dim=16,
        critic_state_encoder_dim=8, critic_action_encoder_dim=8, critic_num_bins=9,
        latent_dim=16, belief_hidden_dim=16, use_amp=False, use_compile=False,
        obs_drop_terms=(), obs_future_steps=None,
    )
    settings.update(options)
    cfg = OneStageFlashSACVelConfig(**settings)
    return OneStageFlashSACVel(cfg, env.observation_spec, env.action_spec, env.reward_spec, "cpu", env)


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
            nxt[BELIEF_KEY] = td["next", BELIEF_KEY].clone()
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
            carry[BELIEF_KEY][done[:, 0]] = 0
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
    assert result["next", BELIEF_KEY].shape == (4, 16)
    student_only = td.select("policy", "vel_command", "depth", "is_init", "step_count", BELIEF_KEY, "ref_joint_pos_")
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


def test_supervision_recomputes_the_current_teacher_latent():
    policy = make_policy()
    collect(policy)
    sequence = policy.buffer.sample()
    with torch.no_grad():
        end = int(sequence.indices.max()) + 1
        predictions = policy.perception.sequence(
            sequence.depth[:, :end].float() / 100, policy._context(sequence.obs[:, :end]),
            sequence.h0, sequence.is_init[:, :end], sequence.valid[:, :end], sequence.learning_start,
        )[0]
        predictions = policy._gather(predictions, sequence.indices)
        raw = policy._gather(sequence.obs, sequence.indices)[..., :policy.replay_obs_dim]
        old_target = policy.actor.encode_priv(raw.flatten(0, 1)).reshape_as(predictions)
        policy.actor.encoder_priv[-1].bias.add_(2.0)
        current_target = policy.actor.encode_priv(raw.flatten(0, 1)).reshape_as(predictions)
        expected_loss = ((predictions - current_target).square().mean(-1) * sequence.sample_weight).sum()
    assert not torch.allclose(old_target, current_target)
    info = policy._train_perception(sequence)
    torch.testing.assert_close(info["perception/priv_loss"], expected_loss)


def test_terminal_bootstrap_forks_final_observation_from_pre_reset_belief():
    policy = make_policy()
    collect(policy)
    sequence = policy.buffer.gather([4, 4], [0, 2]).to("cpu")
    terminal = sequence.terminal_mask & sequence.learning_mask
    assert terminal.any()
    with torch.no_grad():
        _, _, states, _ = policy.perception.sequence(
            sequence.depth.float() / 100, policy._context(sequence.obs), sequence.h0,
            sequence.is_init, sequence.valid,
        )
        expected_hidden = policy._gather(states, sequence.bootstrap_index)[terminal]
    captured = []

    def capture(module, args):
        captured.append(tuple(value.detach().clone() for value in args))

    hook = policy.perception.register_forward_pre_hook(capture)
    batch = policy._sac_batch(sequence)
    hook.remove()
    assert len(captured) == 1
    actual_depth, actual_context, actual_hidden, actual_reset = captured[0]
    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_depth, sequence.terminal_depth[terminal].float() / 100)
    torch.testing.assert_close(actual_context, policy._context(sequence.terminal_obs[terminal]))
    assert not actual_reset.any()
    assert batch["action"].shape[0] == policy.cfg.sample_batch_size
    # FakeEnv's command[0] % 4 encodes environment identity for valid states.
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


def test_teacher_checkpoint_warmstart_preserves_fresh_student_and_perception():
    student = make_policy()
    kwargs = {field.name: getattr(student.cfg, field.name) for field in fields(FlashSACVelConfig)}
    kwargs.update(adapt_module="gru", enable_residual_distillation=False)
    cfg = FlashSACVelConfig(**kwargs)
    env = student.env
    teacher = FlashSACVel(cfg, env.observation_spec, env.action_spec, env.reward_spec, "cpu", env)
    teacher_state = copy.deepcopy(teacher.state_dict())
    before_student = copy.deepcopy(student.student_actor.state_dict())
    before_perception = copy.deepcopy(student.perception.state_dict())
    student.load_state_dict(teacher_state)
    for name in ("actor", "critic", "target_critic", "temperature"):
        assert_nested_equal(teacher_state[name], student.state_dict()[name])
    assert_nested_equal(before_student, student.student_actor.state_dict())
    assert_nested_equal(before_perception, student.perception.state_dict())
    assert student.buffer is None
    assert student._perception_updates == student._update_step == 0
    student._interaction_steps = 1
    state_before_rejected_load = copy.deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="fresh"):
        student.load_state_dict(teacher_state)
    assert_nested_equal(state_before_rejected_load, student.state_dict())
