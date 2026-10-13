"""Full gated perception recurrence and supervised-only gradient tests on CPU."""

import torch
from torch import nn

from active_adaptation.learning.flashsac.one_stage_perception import GatedBeliefPerception


torch.set_num_threads(2)


def make_perception(geometry=None):
    torch.manual_seed(103)
    if geometry is None:
        geometry = torch.randn(12, 3)
    return GatedBeliefPerception(
        context_dim=9,
        depth_shape=(1, 32, 64),
        object_dim=12,
        geometry=geometry,
        latent_dim=16,
        hidden_dim=16,
        depth_feature_dim=8,
    )


def make_sequence(batch_size=3, steps=9):
    return (
        torch.rand(batch_size, steps, 1, 32, 64),
        torch.randn(batch_size, steps, 9),
        torch.randn(batch_size, 16),
        torch.zeros(batch_size, steps, dtype=torch.bool),
        torch.ones(batch_size, steps, dtype=torch.bool),
    )


def test_batched_sequence_matches_online_steps_with_resets_and_placeholders():
    model = make_perception()
    depth, context, hidden, reset, valid = make_sequence()
    reset[0, 0] = True
    reset[1, 4] = True
    reset[2, 7] = True
    valid[2, :2] = False
    valid[0, 5] = False
    with torch.no_grad():
        sequence = model.sequence(depth, context, hidden.clone(), reset, valid)
        outputs = [[], [], [], []]
        for t in range(depth.shape[1]):
            result = model(depth[:, t], context[:, t], hidden, reset[:, t], valid[:, t])
            hidden = result[2]
            for output, value in zip(outputs, result):
                output.append(value)
        online = tuple(torch.stack(output, 1) for output in outputs)
    for batched, sequential in zip(sequence, online):
        torch.testing.assert_close(batched, sequential, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sequence[2][~valid], torch.zeros_like(sequence[2][~valid]))
    assert sequence[0].shape == (3, 9, 16)
    assert sequence[1].shape == (3, 9, 12)
    assert torch.all((sequence[3] >= 0) & (sequence[3] <= 1))


def test_reset_is_applied_before_gate_so_previous_episode_cannot_leak():
    model = make_perception()
    depth = torch.rand(2, 1, 32, 64)
    context = torch.randn(2, 9)
    p, v = model.encode(depth, context)
    old_hidden = torch.randn(2, 16) * 20
    reset = torch.ones(2, dtype=torch.bool)
    from_old = model.step(p, v, old_hidden, reset)
    from_zero = model.step(p, v, torch.zeros_like(old_hidden), reset)
    for actual, expected in zip(from_old, from_zero):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # This also covers invalid replay prefixes: their hidden input is ignored
    # before the gate, and their output hidden is zero rather than a new state.
    invalid = torch.zeros(2, dtype=torch.bool)
    invalid_old = model.step(p, v, old_hidden, ~reset, invalid)
    invalid_zero = model.step(p, v, torch.zeros_like(old_hidden), ~reset, invalid)
    torch.testing.assert_close(invalid_old[3], invalid_zero[3], rtol=0, atol=0)
    torch.testing.assert_close(invalid_old[2], torch.zeros_like(old_hidden), rtol=0, atol=0)
    no_reset = model.step(p, v, old_hidden, ~reset)
    assert not torch.allclose(no_reset[3], from_zero[3])


def test_joint_supervised_loss_reaches_both_heads_and_entire_belief_model():
    model = make_perception()
    depth, context, hidden, reset, valid = make_sequence(batch_size=2, steps=4)
    latent, objects, _, _ = model.sequence(depth, context, hidden, reset, valid)
    teacher_latent = torch.randn_like(latent, requires_grad=True)
    object_target = torch.randn_like(objects)
    loss = nn.functional.mse_loss(latent, teacher_latent.detach()) + nn.functional.mse_loss(objects, object_target)
    loss.backward()
    for branch in ("depth_cnn", "visual", "context", "gate", "gru", "object_head", "priv_head"):
        parameters = list(getattr(model, branch).parameters())
        assert all(parameter.grad is not None for parameter in parameters), branch
        assert sum(parameter.grad.abs().sum().item() for parameter in parameters) > 0, branch
    assert teacher_latent.grad is None


def test_detached_student_actor_loss_does_not_train_perception():
    model = make_perception()
    actor = nn.Linear(9 + 16, 3)
    depth, context, hidden, reset, valid = make_sequence(batch_size=2, steps=3)
    latent = model.sequence(depth, context, hidden, reset, valid)[0]
    actor_input = torch.cat([context, latent.detach()], -1)
    actions = actor(actor_input)
    # A differentiable Q surrogate; its action gradient must reach the actor,
    # while the detach prevents reaching any perception parameter.
    actor_loss = -torch.tanh(actions).mean()
    actor_loss.backward()
    assert actor.weight.grad is not None and actor.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_variable_burn_in_blocks_only_prefix_gradients_and_preserves_values():
    model = make_perception()
    depth, context, hidden, reset, valid = make_sequence()
    depth.requires_grad_()
    context.requires_grad_()
    hidden.requires_grad_()
    learning_start = torch.tensor([2, 4, 6])
    learning = torch.arange(depth.shape[1])[None, :] >= learning_start[:, None]
    with torch.no_grad():
        reference = model.sequence(depth, context, hidden, reset, valid)
    outputs = model.sequence(depth, context, hidden, reset, valid, learning_start)
    for actual, expected in zip(outputs, reference):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    latent_loss = (outputs[0].square().mean(-1) * learning).sum()
    object_loss = (outputs[1].square().mean(-1) * learning).sum()
    (latent_loss + object_loss).backward()
    assert depth.grad is not None and context.grad is not None
    for row, start in enumerate(learning_start.tolist()):
        assert torch.count_nonzero(depth.grad[row, :start]) == 0
        assert torch.count_nonzero(context.grad[row, :start]) == 0
        assert depth.grad[row, start:].abs().sum() > 0
        assert context.grad[row, start:].abs().sum() > 0
    assert hidden.grad is None or torch.count_nonzero(hidden.grad) == 0


def test_fixed_object_frame_geometry_is_detached_cloned_and_not_optimized():
    geometry = torch.randn(12, 3, requires_grad=True)
    expected = geometry.detach().flatten().clone()
    model = make_perception(geometry)
    assert "geometry" in dict(model.named_buffers())
    assert "geometry" not in dict(model.named_parameters())
    assert not model.geometry.requires_grad
    with torch.no_grad():
        geometry.add_(100)
    torch.testing.assert_close(model.geometry, expected, rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    depth, context, hidden, reset, valid = make_sequence(batch_size=2, steps=3)
    latent, objects, _, _ = model.sequence(depth, context, hidden, reset, valid)
    (latent.square().mean() + objects.square().mean()).backward()
    optimizer.step()
    torch.testing.assert_close(model.geometry, expected, rtol=0, atol=0)
    assert geometry.grad is None
