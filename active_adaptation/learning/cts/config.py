"""Independent Hydra options for concurrent teacher/student PPO."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from ..ppo.ppo_vel import PPOConfig


@dataclass
class CTSSettings:
    # User's 512/512 split; the paper's 3:1 split is teacher_ratio=0.75.
    teacher_ratio: float = 0.5
    reconstruction_epochs: int = 2
    student_lr: float = 3e-4
    object_loss_coef: float = 1.0
    normalize_latent: bool = True
    eval_policy: str = "student"
    evaluate_at_end: bool = True


@dataclass
class CTSPPOConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.cts.algorithm.CTSPPOVEL"
    name: str = "cts_ppo_vel"
    runner: str = "cts"
    phase: str = "train"
    vecnorm: str = "train"
    load_noise_scale: float | None = None
    in_keys: list[str] = field(default_factory=lambda: [
        "command", "policy", "priv", "object_", "object_geo_",
        "ref_joint_pos_", "vel_command", "depth", "height",
    ])
    cts: CTSSettings = field(default_factory=CTSSettings)


ConfigStore.instance().store(name="cts_ppo_vel_train", node=CTSPPOConfig, group="algo")
