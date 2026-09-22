"""Hydra entry point for GPO-clip (default) and GPO-penalty."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from ..ppo.ppo_vel import PPOConfig


@dataclass
class GPOSettings:
    variant: str = "clip"
    inner_clip: float = 0.1
    learner_rl_coef: float = 2.0
    # Penalty variant: adaptive alpha multiplies BOTH guider KL and learner RL.
    target_kl: float = 0.001
    penalty_init: float = 0.0
    penalty_factor: float = 1.2
    penalty_max: float = 10.0
    normalize_latent: bool = True
    eval_policy: str = "learner"
    evaluate_at_end: bool = True


@dataclass
class GPOPPOConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.gpo.algorithm.GPOPPOVEL"
    name: str = "gpo_ppo_vel"
    runner: str = "gpo"
    phase: str = "train"
    vecnorm: str = "train"
    load_noise_scale: float | None = None
    # Reference GPO uses a fixed learning rate. Optional PPO KL adaptation uses
    # old-guider/current-guider KL, NOT the guider/learner alignment KL.
    desired_kl: float | None = None
    in_keys: list[str] = field(default_factory=lambda: [
        "command", "policy", "priv", "object_", "object_geo_",
        "ref_joint_pos_", "vel_command", "depth",
    ])
    gpo: GPOSettings = field(default_factory=GPOSettings)


ConfigStore.instance().store(name="gpo_ppo_vel_train", node=GPOPPOConfig, group="algo")
