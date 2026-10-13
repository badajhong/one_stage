"""Joint FlashSAC updates for two policies and one distributional twin critic.

The replay provides an already accumulated n-step reward and its effective
bootstrap discount.  In particular, this module must not multiply by gamma or
mask terminal transitions a second time.  Each policy's categorical Bellman
target is projected separately, then the two distributions are averaged.
"""

from typing import Any, Optional

import torch
from torch.amp.grad_scaler import GradScaler

from .flashsac_upstream.update import update_actor, update_temperature
from .flashsac_upstream.utils_network import Network


def _categorical_target(
    log_probs: torch.Tensor,
    reward: torch.Tensor,
    bootstrap_discount: torch.Tensor,
    entropy_term: torch.Tensor,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """Project a categorical target with an effective discount per transition.

    ``entropy_term`` is alpha * log pi, as in upstream FlashSAC.  Projection is
    done in float32 even under AMP to avoid rounding support indices in FP16.
    A zero discount projects only the accumulated reward, irrespective of the
    action distribution at the subsequent (possibly reset) observation.
    """
    log_probs = log_probs.float()
    num_bins = log_probs.shape[-1]
    if num_bins < 2 or max_v <= min_v:
        raise ValueError("Categorical critic requires at least two bins and max_v > min_v")
    support = torch.linspace(min_v, max_v, num_bins, device=log_probs.device)
    values = reward.float().reshape(-1, 1) + bootstrap_discount.float().reshape(-1, 1) * (
        support[None, :] - entropy_term.float().reshape(-1, 1)
    )
    positions = ((values.clamp(min_v, max_v) - min_v) / (max_v - min_v) * (num_bins - 1))
    positions = positions.clamp(0, num_bins - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp_max(num_bins - 1)
    fraction = positions - lower.float()
    probabilities = log_probs.exp()
    target = torch.zeros_like(probabilities)
    target.scatter_add_(1, lower, probabilities * (1 - fraction))
    target.scatter_add_(1, upper, probabilities * fraction)
    return target


def _min_q_log_probs(qs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """Select the complete distribution from the twin with smaller mean Q."""
    indices = qs.argmin(dim=0)[None, :, None].expand(1, -1, log_probs.shape[-1])
    return log_probs.gather(0, indices)[0]


def _policy_update(
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: dict[str, torch.Tensor],
    cfg: Any,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    # Upstream update_actor freezes critic parameters and uses training=False
    # for its forward, preserving dQ/da without updating critic BatchNorm.
    # Restore flags as well if a caller intentionally froze some Q parameters.
    critic_parameters = list(critic.network.parameters())
    requires_grad = [parameter.requires_grad for parameter in critic_parameters]
    try:
        info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )
    finally:
        for parameter, enabled in zip(critic_parameters, requires_grad):
            parameter.requires_grad_(enabled)
    info.update(update_temperature(temperature, info["actor/entropy"], cfg.temp_target_entropy))
    return info


def update_joint_networks(
    batch: dict[str, torch.Tensor],
    teacher: Network,
    student: Network,
    critic: Network,
    target_critic: Network,
    teacher_temperature: Network,
    student_temperature: Network,
    cfg: Any,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update both policies, one shared twin Q, and its target exactly once.

    Required batch fields are ``observation``, ``next_observation`` (privileged
    Q inputs), ``action`` (normalized tanh action), ``reward``,
    ``bootstrap_discount``, ``actor_observation``, ``actor_next_observation``
    (raw teacher inputs), ``student_observation``, and
    ``student_next_observation`` (context concatenated with student latent).
    Student inputs are detached here too: SAC must never train perception.

    As in upstream FlashSAC, actors and temperatures update before the critic.
    Both actors use the same mixed teacher/student replay batch.  The shared Q
    evaluates continuation under the equally weighted policy mixture.
    """
    device = torch.device(device)
    student_batch = dict(batch)
    student_batch["actor_observation"] = batch["student_observation"].detach()
    student_batch["actor_next_observation"] = batch["student_next_observation"].detach()
    info: dict[str, torch.Tensor] = {}
    if do_actor_update:
        teacher_info = _policy_update(
            teacher, critic, teacher_temperature, batch, cfg, device, grad_scaler
        )
        student_info = _policy_update(
            student, critic, student_temperature, student_batch, cfg, device, grad_scaler
        )
        info.update({f"teacher/{key}": value for key, value in teacher_info.items()})
        info.update({f"student/{key}": value for key, value in student_info.items()})

    batch_size = batch["action"].shape[0]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
        with torch.no_grad():
            teacher_action, teacher_info = teacher(
                observations=batch["actor_next_observation"], training=False
            )
            teacher_action = teacher_action.clone()
            teacher_entropy = teacher_temperature() * teacher_info["log_prob"].clone()
            student_action, student_info = student(
                observations=student_batch["actor_next_observation"], training=False
            )
            student_action = student_action.clone()
            student_entropy = student_temperature() * student_info["log_prob"].clone()

            # Duplicate the current rows to retain upstream FlashSAC's 1:1
            # current/next BatchNorm weighting while giving each policy half
            # the next-state weight.  This is one training forward, not two
            # separate Q updates with policy-dependent running statistics.
            observations = torch.cat([
                batch["observation"], batch["observation"],
                batch["next_observation"], batch["next_observation"],
            ])
            actions = torch.cat([batch["action"], batch["action"], teacher_action, student_action])
            target_qs, target_info = target_critic(
                observations=observations, actions=actions, training=True
            )
            next_qs = target_qs[:, 2 * batch_size:]
            next_log_probs = target_info["log_prob"][:, 2 * batch_size:]
            selected = _min_q_log_probs(next_qs, next_log_probs)
            teacher_target = _categorical_target(
                selected[:batch_size], batch["reward"], batch["bootstrap_discount"],
                teacher_entropy, cfg.critic_min_v, cfg.critic_max_v,
            )
            student_target = _categorical_target(
                selected[batch_size:], batch["reward"], batch["bootstrap_discount"],
                student_entropy, cfg.critic_min_v, cfg.critic_max_v,
            )
            target_probs = (teacher_target + student_target) * 0.5

        _, prediction_info = critic(observations=observations, actions=actions, training=True)
        predicted_log_probs = prediction_info["log_prob"][:, :batch_size].float()
        critic_loss = -(target_probs.unsqueeze(0) * predicted_log_probs).sum(dim=-1).mean()

    assert critic.optimizer is not None
    critic.optimizer.zero_grad(set_to_none=True)
    if cfg.use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(critic_loss).backward()
        grad_scaler.step(critic.optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()
        critic.optimizer.step()
    if critic.scheduler is not None:
        critic.scheduler.step()
    critic.normalize_parameters()
    target_critic.ema_update_parameters()
    info.update({
        "critic/loss": critic_loss,
        "critic/max_entropy_bonus": torch.maximum(teacher_entropy.max(), student_entropy.max()),
    })
    return {key: value.detach() for key, value in info.items()}
