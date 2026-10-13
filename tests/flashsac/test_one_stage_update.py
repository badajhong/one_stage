"""CPU tests for the real FlashSAC networks without importing a simulator."""

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


# The top-level application eagerly imports simulator-dependent algorithms.
# Load only this package and the unmodified vendored FlashSAC modules.
PACKAGE = "_one_stage_update_test_package"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "active_adaptation/learning/flashsac")]
    sys.modules[PACKAGE] = package
update = importlib.import_module(f"{PACKAGE}.one_stage_update")
networks = importlib.import_module(f"{PACKAGE}.flashsac_upstream.network")
Network = importlib.import_module(f"{PACKAGE}.flashsac_upstream.utils_network").Network


def make_network(module, source=None):
    return Network(
        module,
        optimizer=torch.optim.Adam(module.parameters(), lr=0.01) if source is None else None,
        use_weight_normalization=True,
        ema_source=source,
        ema_tau=0.1 if source is not None else None,
    )


def make_fixture():
    torch.manual_seed(27)
    teacher = make_network(networks.FlashSACActor(1, 5, 16, 2))
    student = make_network(networks.FlashSACActor(1, 7, 16, 2))
    critic = make_network(networks.FlashSACDoubleCritic(1, 6, 16, 9, -4, 4))
    target = make_network(copy.deepcopy(critic.network), source=critic)
    teacher_temp = make_network(networks.FlashSACTemperature(0.1))
    student_temp = make_network(networks.FlashSACTemperature(0.2))
    for network in (teacher, student, critic, target):
        network.normalize_parameters()
    batch_size = 8
    batch = {
        "observation": torch.randn(batch_size, 4),
        "next_observation": torch.randn(batch_size, 4),
        "action": torch.rand(batch_size, 2) * 2 - 1,
        "reward": torch.randn(batch_size),
        "bootstrap_discount": torch.tensor([0, 0.1, 0.2, 0.5, 0.9, 0.8, 0.7, 0.6]),
        "actor_observation": torch.randn(batch_size, 5),
        "actor_next_observation": torch.randn(batch_size, 5),
        # Deliberately attached to a hypothetical perception graph: the update
        # must enforce the supervised-only perception contract itself.
        "student_observation": torch.randn(batch_size, 7, requires_grad=True),
        "student_next_observation": torch.randn(batch_size, 7, requires_grad=True),
    }
    cfg = SimpleNamespace(
        actor_bc_alpha=0.1, temp_target_entropy=-1.0, use_amp=False,
        critic_min_v=-4.0, critic_max_v=4.0, critic_num_bins=9,
        # These must not be applied again to bootstrap_discount.
        gamma=0.3, n_step=7,
    )
    return batch, teacher, student, critic, target, teacher_temp, student_temp, cfg


def test_categorical_projection_respects_effective_discount_and_terminal():
    # Point mass at Q=2 on support [-2,-1,0,1,2].
    log_probs = torch.full((3, 5), -torch.inf)
    log_probs[:, -1] = 0
    result = update._categorical_target(
        log_probs,
        reward=torch.tensor([0.0, -0.5, 0.0]),
        bootstrap_discount=torch.tensor([0.5, 0.0, 1.0]),
        entropy_term=torch.tensor([0.0, 100.0, -100.0]),
        min_v=-2.0, max_v=2.0,
    )
    expected = torch.tensor([
        [0, 0, 0, 1, 0],        # 0 + .5 * 2 = 1
        [0, .5, .5, 0, 0],     # terminal ignores future Q/entropy
        [0, 0, 0, 0, 1],       # clamp upper endpoint; preserve probability
    ])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.sum(dim=1), torch.ones(3))


def test_joint_target_mixes_distributions_instead_of_averaging_values():
    fixture = make_fixture()
    batch, teacher, student, critic, target, teacher_temp, student_temp, cfg = fixture
    batch["reward"].zero_()
    batch["bootstrap_discount"].fill_(1.0)
    # Force the two projected targets to opposite ends of the support.  The
    # critic loss must see a 50:50 bimodal target, never a point mass at mean=0.
    extremes = [torch.nn.functional.one_hot(torch.zeros(8, dtype=torch.long), 9).float(),
                torch.nn.functional.one_hot(torch.full((8,), 8), 9).float()]
    captured = []

    def capture_prediction(module, args, kwargs, result):
        if kwargs["training"]:
            captured.append(result[1]["log_prob"][:, :8].detach().clone())

    hook = critic.network.register_forward_hook(capture_prediction, with_kwargs=True)
    with patch.object(update, "_categorical_target", side_effect=extremes) as project:
        info = update.update_joint_networks(*fixture, False, torch.device("cpu"), None)
    hook.remove()
    assert project.call_count == 2
    expected_loss = -(0.5 * captured[0][..., 0] + 0.5 * captured[0][..., -1]).mean()
    torch.testing.assert_close(info["critic/loss"], expected_loss)
    # The helper received replay's discount unchanged for both policies.
    for call in project.call_args_list:
        assert call.args[2] is batch["bootstrap_discount"]


def test_joint_update_has_one_q_step_one_ema_and_two_independent_actor_steps():
    fixture = make_fixture()
    batch, teacher, student, critic, target, teacher_temp, student_temp, cfg = fixture
    before = {
        name: [parameter.detach().clone() for parameter in net.network.parameters()]
        for name, net in (("teacher", teacher), ("student", student), ("critic", critic),
                          ("teacher_temp", teacher_temp), ("student_temp", student_temp))
    }
    critic_modes = []
    target_modes = []
    critic_hook = critic.network.register_forward_pre_hook(
        lambda module, args, kwargs: critic_modes.append((kwargs["training"], kwargs["observations"].shape[0])),
        with_kwargs=True,
    )
    target_hook = target.network.register_forward_pre_hook(
        lambda module, args, kwargs: target_modes.append((kwargs["training"], kwargs["observations"].shape[0])),
        with_kwargs=True,
    )
    with patch.object(critic.optimizer, "step", wraps=critic.optimizer.step) as q_step, \
            patch.object(target, "ema_update_parameters", wraps=target.ema_update_parameters) as ema, \
            patch.object(teacher.optimizer, "step", wraps=teacher.optimizer.step) as teacher_step, \
            patch.object(student.optimizer, "step", wraps=student.optimizer.step) as student_step:
        info = update.update_joint_networks(*fixture, True, torch.device("cpu"), None)
        assert q_step.call_count == ema.call_count == teacher_step.call_count == student_step.call_count == 1
    critic_hook.remove()
    target_hook.remove()
    assert critic_modes == [(False, 8), (False, 8), (True, 32)]
    assert target_modes == [(True, 32)]
    for name, net in (("teacher", teacher), ("student", student), ("critic", critic),
                      ("teacher_temp", teacher_temp), ("student_temp", student_temp)):
        assert any(not torch.equal(old, new) for old, new in zip(before[name], net.network.parameters())), name
    assert batch["student_observation"].grad is None
    assert batch["student_next_observation"].grad is None
    assert all(parameter.requires_grad for parameter in critic.network.parameters())
    assert all(torch.isfinite(value).all() and not value.requires_grad for value in info.values())
    assert "teacher/actor/loss" in info and "student/actor/loss" in info


def test_delayed_actor_step_leaves_both_policies_and_temperatures_unchanged():
    fixture = make_fixture()
    batch, teacher, student, critic, target, teacher_temp, student_temp, cfg = fixture
    policies = [teacher, student, teacher_temp, student_temp]
    before = [[parameter.detach().clone() for parameter in net.network.parameters()] for net in policies]
    update.update_joint_networks(*fixture, False, torch.device("cpu"), None)
    for net, old_parameters in zip(policies, before):
        for old, new in zip(old_parameters, net.network.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_actor_updates_do_not_accumulate_q_parameter_gradients_or_bn_statistics():
    batch, teacher, student, critic, target, teacher_temp, student_temp, cfg = make_fixture()
    running_before = {name: value.clone() for name, value in critic.network.named_buffers()}
    info = update._policy_update(teacher, critic, teacher_temp, batch, cfg, torch.device("cpu"), None)
    assert all(parameter.grad is None for parameter in critic.network.parameters())
    for name, value in critic.network.named_buffers():
        torch.testing.assert_close(running_before[name], value, rtol=0, atol=0)
    assert torch.isfinite(info["actor/loss"])
