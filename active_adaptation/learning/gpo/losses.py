"""Equations (11)/(12) of arXiv:2505.15418, with explicit gradient boundaries.

Adapted from the authors' GPO.py; see LICENSE.GPO for its MIT notice.

Likelihood ratios use joint action log-probabilities. Alignment KL is averaged
over action dimensions, matching the authors' GPO.py (rather than Torch's
Independent KL sum). PPO old/new KL diagnostics still use the joint KL.
"""

import math

import torch
from torch.distributions import kl_divergence

from ..modules.distributions import IndependentNormal


def masked_mean(value, valid):
    return torch.where(valid, value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(1)


def policy_losses(guider, learner, action, old_log_prob, advantage, valid,
                  clip_param, settings, penalty_coef=0.0):
    """Both branches reuse actions sampled from beta = old guider.

    Stop gradients on distribution *outputs*, including scale: a shared actor
    receives gradients from both live branches in one joint optimizer step.
    """
    beta_log = old_log_prob.detach()
    advantage = advantage.detach()
    g_log = guider.log_prob(action)
    l_log = learner.log_prob(action)
    frozen_guider = IndependentNormal(guider.loc.detach(), guider.scale.detach())
    frozen_learner = IndependentNormal(learner.loc.detach(), learner.scale.detach())
    action_dim = action.shape[-1]
    learner_kl = kl_divergence(frozen_guider, learner) / action_dim
    guider_kl = kl_divergence(guider, frozen_learner) / action_dim

    # Guard pathological exponentials; ordinary PPO ratios are unaffected.
    # Learner cap=10 is the numerical safeguard in the reference implementation.
    g_ratio = (g_log - beta_log).clamp(max=60).exp()
    l_ratio = (l_log - beta_log).clamp(max=10).exp()
    low, high = math.log1p(-clip_param), math.log1p(clip_param)
    if settings.variant == "clip":
        inner_low = math.log1p(-settings.inner_clip)
        inner_high = math.log1p(settings.inner_clip)
        log_relative = g_log - l_log.detach()
        # Clip in log space before exp, avoiding overflow in the outer clip.
        clipped_log = log_relative.clamp(inner_low, inner_high) + l_log.detach() - beta_log
        g_clipped = clipped_log.clamp(low, high).exp()
        outside = ((log_relative.detach() < inner_low) | (log_relative.detach() > inner_high))
        backtrack = masked_mean(guider_kl * outside, valid)
        rl_coef = settings.learner_rl_coef
    elif settings.variant == "penalty":
        g_clipped = g_ratio.clamp(1 - clip_param, 1 + clip_param)
        outside = torch.zeros_like(valid)
        backtrack = penalty_coef * masked_mean(guider_kl, valid)
        rl_coef = penalty_coef
    else:
        raise ValueError(f"Unknown GPO variant: {settings.variant}")
    l_clipped = l_ratio.clamp(1 - clip_param, 1 + clip_param)
    guider_ppo = -masked_mean(torch.minimum(g_ratio * advantage, g_clipped * advantage), valid)
    learner_ppo = -masked_mean(torch.minimum(l_ratio * advantage, l_clipped * advantage), valid)
    imitation = masked_mean(learner_kl, valid)
    return {
        "guider_ppo": guider_ppo,
        "backtrack": backtrack,
        "learner_ppo": learner_ppo,
        "imitation": imitation,
        "policy": guider_ppo + backtrack + imitation + rl_coef * learner_ppo,
        "alignment_kl": imitation.detach(),
        "backtrack_fraction": masked_mean(outside.float(), valid).detach(),
        "guider_clip_fraction": masked_mean(((g_ratio - 1).abs() > clip_param).float(), valid).detach(),
        "learner_clip_fraction": masked_mean(((l_ratio - 1).abs() > clip_param).float(), valid).detach(),
    }


def adapt_penalty(coef, distance, settings):
    """Reference epoch-wise adaptive alpha, including startup from zero."""
    if coef < 1e-5 and distance > settings.target_kl:
        coef = 0.1
    if distance > settings.target_kl * settings.penalty_factor:
        coef *= settings.penalty_factor
    elif distance < settings.target_kl / settings.penalty_factor:
        coef /= settings.penalty_factor
    return min(coef, settings.penalty_max)
