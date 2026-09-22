"""SITT objectives. Environment reward tensors never enter these functions in-place."""

from contextlib import contextmanager

import torch
from torch.distributions import kl_divergence

from ..modules.distributions import IndependentNormal


def action_kl(loc, scale, target_loc, target_scale, max_kl=None):
    """KL(policy || frozen target), summed over joints; output [..., 1]."""
    dist = IndependentNormal(loc, scale)
    target = IndependentNormal(target_loc.detach(), target_scale.detach())
    kl = kl_divergence(dist, target).unsqueeze(-1)
    if not torch.isfinite(kl).all():
        raise FloatingPointError("Non-finite SITT action KL; inspect action means/std.")
    return kl if max_kl is None else kl.clamp_max(max_kl)


def shared_covariance_kl(loc, target_loc, scale, max_kl=None):
    """SITT Eq. (10), with a frozen target mean and ONE live shared std.

    Detaching a separate target covariance in a generic Gaussian KL would
    remove the mean-mismatch gradient into the shared covariance. Here the
    exact equal-covariance expression retains d KL / d sigma = -delta^2/sigma^3.
    """
    kl = 0.5 * ((loc - target_loc.detach()) / scale).square().sum(-1, keepdim=True)
    if not torch.isfinite(kl).all():
        raise FloatingPointError("Non-finite SITT shared-covariance KL; inspect action means/std.")
    return kl if max_kl is None else kl.clamp_max(max_kl)


def ramp_factor(num_updates, warmup_iters, ramp_iters):
    if num_updates < warmup_iters:
        return 0.0
    if ramp_iters == 0:
        return 1.0
    return min(1.0, (num_updates - warmup_iters) / ramp_iters)


@contextmanager
def frozen(module):
    """Freeze decoder weights while retaining derivatives w.r.t. its input."""
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)
