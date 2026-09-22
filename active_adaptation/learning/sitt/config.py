"""Hydra registration for the VAIC adaptation of SITT."""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from ..ppo.ppo_vel import PPOConfig


@dataclass
class SITTSettings:
    # These are starting values, not tuned skateboard results.
    reward_kl_coef: float = 0.01
    policy_kl_coef: float = 0.01
    warmup_iters: int = 100
    ramp_iters: int = 200
    alignment_epochs: int = 2
    proxy_epochs: int = 2
    proxy_lr: float = 3e-4
    proxy_action_coef: float = 1.0
    student_action_coef: float = 1.0
    ema_tau: float = 0.04
    # Match the environment's per-second reward convention, without changing it.
    kl_reward_mult_dt: bool = True
    # None means the exact diagonal Gaussian KL, summed over action dimensions.
    max_kl: float | None = None
    eval_policy: str = "student"
    evaluate_at_end: bool = True


@dataclass
class SITTPPOConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.sitt.algorithm.SITTPPOVEL"
    name: str = "sitt_ppo_vel"
    runner: str = "sitt"
    phase: str = "train"
    vecnorm: str = "train"
    # Exact SITT checkpoint restoration must not replace learned exploration std.
    load_noise_scale: float | None = None
    in_keys: list[str] = field(default_factory=lambda: [
        "command", "policy", "priv", "object_", "object_geo_",
        "ref_joint_pos_", "vel_command", "depth", "height",
    ])
    sitt: SITTSettings = field(default_factory=SITTSettings)


ConfigStore.instance().store(
    name="sitt_ppo_vel_train", node=SITTPPOConfig, group="algo"
)
