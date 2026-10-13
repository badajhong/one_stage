"""Finite-window causality, cache freshness and supervised gradient contracts."""

import pytest
import torch
from torch import nn
from active_adaptation.learning.flashsac.one_stage_perception import (
    CausalWindowBlock, WindowTransformerPerception, WindowHistory,
)

torch.set_num_threads(2)


def make_perception(geometry=None, **kwargs):
    torch.manual_seed(103)
    if geometry is None:
        geometry = torch.randn(12, 3)
    return WindowTransformerPerception(
        context_dim=9, depth_shape=(1, 32, 64), object_dim=12, geometry=geometry,
        latent_dim=16, window_size=kwargs.pop('window_size', 4), model_dim=16,
        num_heads=4, ffn_dim=32, depth_feature_dim=8, **kwargs)


def make_sequence(batch_size=3, steps=9):
    return (torch.randint(101, (batch_size, steps, 1, 32, 64)).float() / 100,
            torch.randn(batch_size, steps, 9),
            torch.zeros(batch_size, steps, dtype=torch.bool),
            torch.ones(batch_size, steps, dtype=torch.bool))


@pytest.mark.parametrize('layers', [1, 2])
@pytest.mark.parametrize('length', [1, 4])
def test_replay_windows_match_online_ring_with_resets_placeholders_and_wrap(layers, length):
    model = make_perception(num_layers=layers, window_size=length)
    depth, context, reset, valid = make_sequence(steps=11)
    reset[0, 0] = True
    reset[1, 4] = True
    reset[2, 7] = True
    valid[2, :2] = False
    valid[0, 5] = False
    history = WindowHistory(model, 3, 100)
    with torch.no_grad():
        sequence = model.sequence(depth, context, reset, valid)
        online = [history.step(model, depth[:, t], context[:, t], reset[:, t], valid[:, t], 0)
                  for t in range(depth.shape[1])]
        for component in (0, 1):
            stacked = torch.stack([item[component] for item in online], 1)
            torch.testing.assert_close(stacked[:, length-1:], sequence[component], rtol=1e-5, atol=1e-6)
        for t in range(length-1, depth.shape[1]):
            start = t-length+1
            reference = model(depth[:, start:t+1], context[:, start:t+1],
                              reset[:, start:t+1], valid[:, start:t+1])
            for a, b in zip(reference, online[t]):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(sequence[0][~valid[:, length-1:]]) == 0


def test_masked_frames_and_future_frames_cannot_affect_values_or_gradients():
    model = make_perception(num_layers=2)
    depth, context, reset, valid = make_sequence(batch_size=2, steps=9)
    reset[:, 3] = True
    depth.requires_grad_(); context.requires_grad_()
    output = model.sequence(depth, context, reset, valid)[0][:, 1]  # query t=4
    output.square().sum().backward()
    assert depth.grad[:, 3:5].abs().sum() > 0
    assert context.grad[:, 3:5].abs().sum() > 0
    assert not depth.grad[:, :3].any() and not depth.grad[:, 5:].any()
    assert not context.grad[:, :3].any() and not context.grad[:, 5:].any()
    with torch.no_grad():
        changed = depth.clone(); changed[:, :3] += 100; changed[:, 5:] += 100
        prediction = model.sequence(changed, context, reset, valid)[0][:, 1]
    torch.testing.assert_close(prediction, output)


def test_context_prefix_receives_supervised_gradient_without_burnin():
    model = make_perception()
    depth, context, reset, valid = make_sequence(batch_size=2, steps=4)
    depth.requires_grad_(); context.requires_grad_()
    z, obj = model.sequence(depth, context, reset, valid)
    target = torch.randn_like(z, requires_grad=True)
    (nn.functional.mse_loss(z, target.detach()) + obj.square().mean()).backward()
    assert depth.grad[:, :3].abs().sum() > 0 and context.grad[:, :3].abs().sum() > 0
    for branch in ('depth_cnn', 'visual', 'context', 'token', 'blocks', 'object_head', 'priv_head'):
        params = list(getattr(model, branch).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(p.grad.abs().sum().item() for p in params) > 0
    assert target.grad is None


def test_last_query_optimization_preserves_outputs_and_gradients():
    block = CausalWindowBlock(16, 4, 32)
    x = torch.randn(3, 4, 16, requires_grad=True)
    valid = torch.tensor([[False, False, True, True], [True]*4, [False]*4])
    full = block(x, valid)[:, -1:]
    last = block(x, valid, last_only=True)
    torch.testing.assert_close(full, last, rtol=1e-5, atol=1e-6)
    inputs = [x, *block.parameters()]
    a = torch.autograd.grad(full.square().sum(), inputs, retain_graph=True)
    b = torch.autograd.grad(last.square().sum(), inputs)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)


def test_weight_change_refreshes_all_cached_frames_with_current_encoder():
    model = make_perception()
    history = WindowHistory(model, 3, 100)
    depth, context, reset, valid = make_sequence(steps=7)
    for t in range(6):
        history.step(model, depth[:, t], context[:, t], reset[:, t], valid[:, t], 0)
    with torch.no_grad():
        model.token.weight.add_(torch.randn_like(model.token.weight) * .1)
        actual = history.step(model, depth[:, 6], context[:, 6], reset[:, 6], valid[:, 6], 1)
        expected = model(depth[:, 3:7], context[:, 3:7], reset[:, 3:7], valid[:, 3:7])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert history.version == 1


def test_invalid_windows_are_finite_and_zero_and_reset_clears_history():
    model = make_perception()
    depth, context, reset, valid = make_sequence(steps=4)
    valid.zero_()
    z, obj = model(depth, context, reset, valid)
    assert torch.isfinite(z).all() and torch.isfinite(obj).all()
    assert not z.any() and not obj.any()
    (z.sum()+obj.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    history = WindowHistory(model, 3, 100)
    history.valid.fill_(True)
    history.step(model, depth[:, 0], context[:, 0], torch.ones(3, dtype=torch.bool),
                 torch.ones(3, dtype=torch.bool), 0)
    assert torch.equal(history.valid.sum(-1), torch.ones(3, dtype=torch.long))


def test_detached_student_actor_loss_does_not_train_perception():
    model = make_perception()
    actor = nn.Linear(9 + 16, 3)
    depth, context, reset, valid = make_sequence(batch_size=2, steps=6)
    latent = model.sequence(depth, context, reset, valid)[0]
    actions = actor(torch.cat([context[:, 3:], latent.detach()], -1))
    (-torch.tanh(actions).mean()).backward()
    assert actor.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_fixed_geometry_is_detached_cloned_and_not_optimized():
    geometry = torch.randn(12, 3, requires_grad=True)
    expected = geometry.detach().flatten().clone()
    model = make_perception(geometry)
    with torch.no_grad(): geometry.add_(100)
    assert 'geometry' in dict(model.named_buffers())
    assert 'geometry' not in dict(model.named_parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    z, obj = model.sequence(*make_sequence(batch_size=2, steps=4))
    (z.square().mean()+obj.square().mean()).backward(); optimizer.step()
    torch.testing.assert_close(model.geometry, expected, rtol=0, atol=0)
    assert geometry.grad is None
