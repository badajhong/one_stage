import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools
import torch.utils._pytree as pytree
import einops
import copy
import numpy as np

from torchrl.data import CompositeSpec, TensorSpec, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod, 
    TensorDictSequential as Seq
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, List
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.rnn import set_recurrent_mode, recurrent_mode
from .common import *

from active_adaptation.utils.torch_compat import enable_tf32

enable_tf32()

OBJECT_KEY = "object_"
DEPTH_KEY = "depth"       # student depth input (RGB-D camera)
HEIGHT_KEY = "height"     # teacher height-map input (ray-caster grid)
VEL_CMD_KEY = "vel_command"  # student velocity command [vx, vy, vyaw]
REF_JPOS_KEY = "ref_joint_pos_"
PRIV_FEATURE_KEY = "priv_feature"
PRIV_PRED_KEY = "priv_pred"
OBJECT_PRED_KEY = "object_pred"
OBJECT_GEO_KEY = "object_geo_"
OBJECT_TRANS_KEY = "object_trans"
OBJECT_PRED_TRANS_KEY = "object_pred_trans"

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_vel.PPOVEL"
    name: str = "ppo_vel"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    enable_residual_distillation: bool = True
    distill_with_priv_pred: bool = False

    train_dr_estimator: bool = False
    normalize_ratio: bool = False

    # Ablation flags
    use_object_adapt: bool = True   # False → no object state prediction (no object_adapt, no object_trans in adapt_module)
    use_depth: bool = True          # False → depth feature always zero (no temporal_depth_gru even in finetune)

    # lr linear schedule or adaptive lr
    lr: float = 3e-4

    desired_kl: float | None = 0.01 # None

    # entropy coef schedule
    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    entropy_decay_iters: int = 1000

    init_noise_scale: float = 1.0
    load_noise_scale: float | None = 0.5

    clip_neg_reward: bool = False

    normalize_before_sum: bool = False

    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    adapt_module: str = "gru" # "gru", "mlp"
    latent_dim: int = 256
    adapt_module_input_cmd: bool = True

    max_grad_norm: float = 1.0

    clip_adv: float | None = None
    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY)

cs = ConfigStore.instance()
cs.store("ppo_vel_train", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)), group="algo")
# cs.store("ppo_vel_finetune", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001, enable_residual_distillation=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, VEL_CMD_KEY)), group="algo")
cs.store("ppo_vel_finetune", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001, enable_residual_distillation=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, VEL_CMD_KEY, DEPTH_KEY)), group="algo")

# Ablation 1: no object adaptation (no object_adapt, object_trans not fed into adapt_module)
# Ablation: w/o Object State Estimation
cs.store("ppo_vel_train_noobj", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001, use_object_adapt=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)), group="algo")
cs.store("ppo_vel_finetune_noobj", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001, enable_residual_distillation=False, use_object_adapt=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, VEL_CMD_KEY, DEPTH_KEY)), group="algo")

# Ablation 2: no depth (depth feature always zero, even in finetune)
# Ablation: w/o Depth
cs.store("ppo_vel_train_nodepth", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001, use_depth=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)), group="algo")
cs.store("ppo_vel_finetune_nodepth", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001, enable_residual_distillation=False, use_depth=False, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, VEL_CMD_KEY)), group="algo")

# Ablation 3: MLP instead of GRU for adapt_module (no temporal context)
# Ablation: w/o GRU (MLP adapt)
cs.store("ppo_vel_train_nogru", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001, adapt_module="mlp", in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)), group="algo")
cs.store("ppo_vel_finetune_nogru", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001, enable_residual_distillation=False, adapt_module="mlp", in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, VEL_CMD_KEY, DEPTH_KEY)), group="algo")


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, burn_in=True):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if recurrent_mode():
            N, T = x.shape[:2]
            hx = hx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)
        else:
            N = x.shape[0]
            hx = self.gru(x, hx)
            output = self.ln(hx)
            return output, hx


class GRUModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = make_mlp([dim, dim])
        self.gru = GRU(dim, hidden_size=dim)
        self.out = nn.LazyLinear(dim)

    def forward(self, x, is_init, hx):
        out1 = self.mlp(x)
        out2, hx = self.gru(out1, is_init, hx)
        out3 = self.out(out2 + out1)
        return (out3, hx.contiguous())


class ZeroDepthInjector(TensorDictModuleBase):
    """Injects a zero tensor as '_depth_feature' when no depth sensor is available."""
    in_keys = []
    out_keys = ["_depth_feature"]

    def __init__(self, dim: int, dev):
        super().__init__()
        self.dim = dim
        self._dev = dev

    def forward(self, td):
        ref = next(iter(td.values(True, True)), None)
        dev = ref.device if ref is not None else self._dev
        if ref is not None:
            batch_shape = ref.shape[:-1]
            td["_depth_feature"] = torch.zeros(*batch_shape, self.dim, device=dev)
        else:
            td["_depth_feature"] = torch.zeros(*td.batch_size, self.dim, device=dev)
        return td


class TemporalDepthGRU(TensorDictModuleBase):
    """Processes single-frame depth through CNN + GRU for temporal depth feature.

    Input key: DEPTH_KEY [N, 1, H, W]
    Output keys: "_depth_feature" [N, hidden_dim], ("next", "depth_hx") [N, hidden_dim]
    """
    in_keys = [DEPTH_KEY, "is_init", "depth_hx"]
    out_keys = ["_depth_feature", ("next", "depth_hx")]

    def __init__(self, depth_cnn, hidden_dim=64):
        super().__init__()
        self.depth_cnn = depth_cnn
        self.gru = GRU(hidden_dim, hidden_size=hidden_dim, burn_in=False)
        self.out = nn.LayerNorm(hidden_dim)

    def forward(self, tensordict):
        depth_frame = tensordict[DEPTH_KEY]  # [N, 1, H, W]
        is_init = tensordict["is_init"]
        depth_hx = tensordict["depth_hx"]

        # CNN: [N, 1, H, W] -> [N, 64]
        feature = self.depth_cnn(depth_frame)  # FlattenBatch(data_dim=3) handles [N, 1, H, W]
        # GRU: single timestep
        out, depth_hx = self.gru(feature, is_init, depth_hx)
        tensordict["_depth_feature"] = self.out(out)
        tensordict.set(("next", "depth_hx"), depth_hx)
        return tensordict


class TransformObject(TensorDictModuleBase):
    def __init__(self, output_dim, in_keys, out_keys):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.points_dim = 128
        self.transform_dim = 12

    def forward(self, tensordict):
        # Identify keys
        geo_key = OBJECT_GEO_KEY
        # The other key is the vector (pose)
        vec_key = [k for k in self.in_keys if k != geo_key][0]
        
        object_geo_ = tensordict[geo_key].view(*tensordict[geo_key].shape[:-1], -1, 3)
        objects_num = object_geo_.shape[-2] // self.points_dim
        object_vec = tensordict[vec_key][..., -objects_num*self.transform_dim:]
        points_trans = []
        for i in range(objects_num):
            # For mutli objects, we compute each objet separately and concatenate
            # object_pos_b (3) + object_ori_b (9)
            pos = object_vec[..., i*self.transform_dim:i*self.transform_dim+3]
            ori = object_vec[..., i*self.transform_dim+3:i*self.transform_dim+12].view(*object_vec.shape[:-1], 3, 3)
            object_points = object_geo_[..., i*self.points_dim:(i+1)*self.points_dim, :] # B x N x 3
            points_rot = torch.matmul(object_points, ori.transpose(-1, -2)) # B x N x 3
            points_trans.append(points_rot + pos.unsqueeze(-2)) # B x N x 3
        points_trans = torch.cat(points_trans, dim=-2)

        tensordict[self.out_keys[0]] = points_trans.flatten(-2, -1)
        return tensordict


class PPOVEL(TensorDictModuleBase):
    train_in_keys = [CMD_KEY, OBS_KEY, OBS_PRIV_KEY, ACTION_KEY,
                     "adv", "ret", "is_init", "sample_log_prob", "step_count"]
    
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "finetune"]

        self.entropy_coef = self.cfg.entropy_coef_start
        self.desired_kl = cfg.desired_kl
        self.clip_param = self.cfg.clip_param

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.rec_loss = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=self.cfg.gamma, lmbda=self.cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales /= self.reward_scales.sum()
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)
        object.__setattr__(self, "env", env)

        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names
        
        fake_input = observation_spec.zero()

        # Add depth_hx to fake_input for initialization (needed even in train phase for consistency)
        fake_input["depth_hx"] = torch.zeros(fake_input.shape[0], 64)

        if observation_spec.get("command_", None) is not None:
            global CMD_KEY
            CMD_KEY = "command_"
        
        # build encoder, adapt module, critic
        encoder_priv_in_keys = [OBS_PRIV_KEY]
        adapt_module_in_keys = [OBS_KEY]
        critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]
        if self.cfg.adapt_module_input_cmd:
            adapt_module_in_keys.append(VEL_CMD_KEY)
        if observation_spec.get(OBJECT_KEY, None) is not None:
            encoder_priv_in_keys.append(OBJECT_KEY)
            encoder_priv_in_keys.append(OBJECT_TRANS_KEY)
            critic_in_keys.append(OBJECT_KEY)
            if self.cfg.use_object_adapt:
                # adapt_module uses predicted object (finetune) or GT object (train)
                if self.cfg.phase == "train":
                    adapt_module_in_keys.append(OBJECT_KEY)
                    adapt_module_in_keys.append(OBJECT_TRANS_KEY)
                else:
                    adapt_module_in_keys.append(OBJECT_PRED_KEY)
                    adapt_module_in_keys.append(OBJECT_PRED_TRANS_KEY)
            # else: ablation noobj — adapt_module sees no object info

        object_dim = observation_spec[OBJECT_KEY].shape[-1]
        latent_dim = self.cfg.latent_dim
        # For gt object transform
        self.object_transform = Seq(
            TransformObject(object_dim, [OBJECT_KEY, OBJECT_GEO_KEY], [OBJECT_TRANS_KEY]),
        ).to(self.device)
        # For predicted object transform
        self.object_pred_transform = Seq(
            TransformObject(object_dim, [OBJECT_PRED_KEY, OBJECT_GEO_KEY], [OBJECT_PRED_TRANS_KEY]),
        ).to(self.device)
        # use_depth=False forces depth feature to zero even if depth is in obs spec
        has_depth = (observation_spec.get(DEPTH_KEY, None) is not None) and self.cfg.use_depth
        has_height = observation_spec.get(HEIGHT_KEY, None) is not None

        # Teacher height CNN: encodes height-map grid → "_height_feature"
        if has_height:
            self.height_cnn = nn.Sequential(
                make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
                nn.LazyLinear(64),
                nn.LayerNorm(64)
            ).to(self.device)
            self.height_encoder = Mod(self.height_cnn, [HEIGHT_KEY], ["_height_feature"]).to(self.device)

        # Student depth CNN + temporal GRU: processes depth frame sequence → "_depth_feature"
        if has_depth:
            self.depth_cnn = nn.Sequential(
                make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
                nn.LazyLinear(64),
                nn.LayerNorm(64)
            ).to(self.device)
            self.temporal_depth_gru = TemporalDepthGRU(self.depth_cnn, hidden_dim=64).to(self.device)

        self.depth_feature_dim = 64  # depth feature output dim, used for zero injection in train phase

        # object_adapt - keeps depth fusion for object prediction only.
        # Skipped when use_object_adapt=False (ablation: no object state estimation).
        if self.cfg.use_object_adapt:
            self.object_adapt = Seq(
                CatTensors([OBS_KEY, VEL_CMD_KEY], "_object_adapt_mlp_inp", del_keys=False, sort=False),
                Mod(make_mlp([latent_dim]), "_object_adapt_mlp_inp", ["_obj_adapt_mlp"]),
                CatTensors(["_obj_adapt_mlp", "_depth_feature"], "_object_adapt_inp", del_keys=False),
                Mod(nn.Sequential(make_mlp([latent_dim, latent_dim]), nn.LazyLinear(object_dim)),
                    "_object_adapt_inp", OBJECT_PRED_KEY),
                selected_out_keys=[OBJECT_PRED_KEY]
            ).to(self.device)
        # Teacher encoder_priv uses height feature; student/finetune does not use height
        if has_height:
            encoder_priv_in_keys.append("_height_feature")
        self.encoder_priv = Seq(
            CatTensors(encoder_priv_in_keys, "_encoder_priv_inp", del_keys=False, sort=False),
            Mod(nn.Sequential(make_mlp([latent_dim]), nn.LazyLinear(latent_dim)), "_encoder_priv_inp", PRIV_FEATURE_KEY),
            selected_out_keys=[PRIV_FEATURE_KEY]
        ).to(self.device)

        if self.cfg.adapt_module == "gru":
            self.adapt_module = Seq(
                CatTensors(adapt_module_in_keys, "_adapt_inp", del_keys=False, sort=False),
                Mod(GRUModule(latent_dim), ["_adapt_inp", "is_init", "adapt_hx"], [PRIV_PRED_KEY, ("next", "adapt_hx")]),
                selected_out_keys=[PRIV_PRED_KEY, ("next", "adapt_hx")]
            ).to(self.device)
        elif self.cfg.adapt_module == "mlp":
            self.adapt_module = Seq(
                CatTensors(adapt_module_in_keys, "_adapt_inp", del_keys=False, sort=False),
                Mod(nn.Sequential(make_mlp([latent_dim, latent_dim]), nn.LazyLinear(latent_dim)), "_adapt_inp", [PRIV_PRED_KEY]),
                selected_out_keys=[PRIV_PRED_KEY],
            ).to(self.device)
        else:
            raise ValueError(f"Invalid adapt module: {self.cfg.adapt_module}")
        
        # build actor
        if cfg.phase == "train" and cfg.enable_residual_distillation:
            assert REF_JPOS_KEY in observation_spec, f"{REF_JPOS_KEY} should be in observation_spec"
            class RefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return (ref_jpos + action,)
            residual_module_cls = RefJointPos
        else:
            class DummyRefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return action
            residual_module_cls = DummyRefJointPos
        in_keys = [REF_JPOS_KEY, "loc"]
        out_keys = ["loc"]
        residual_module = Mod(residual_module_cls(), in_keys, out_keys)

        def build_actor(in_keys: List[str], dist_cls, dist_keys, residual_module=None) -> ProbabilisticActor:
            actor_modules = [
                    CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                    Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
                    Mod(Actor(self.action_dim, init_noise_scale=self.cfg.init_noise_scale, load_noise_scale=self.cfg.load_noise_scale), ["_actor_feature"], dist_keys)
            ]
            if residual_module is not None:
                actor_modules.append(residual_module)
            actor_module = Seq(*actor_modules)
            actor = ProbabilisticActor(
                module=actor_module,
                in_keys=dist_keys,
                out_keys=[ACTION_KEY],
                distribution_class=dist_cls,
                return_log_prob=True
            ).to(self.device)
            return actor

        self.dist_cls = IndependentNormal
        self.dist_keys = IndependentNormal.dist_keys

        in_keys = [CMD_KEY, OBS_KEY, PRIV_FEATURE_KEY]
        self.actor = build_actor(in_keys, self.dist_cls, self.dist_keys, residual_module=residual_module)
        # Student actor uses velocity command [vx, vy, vyaw] instead of reference motion command
        in_keys = [VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
        self.actor_adapt = build_actor(in_keys, self.dist_cls, self.dist_keys)

        # build critic
        _critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(num_reward_groups))
        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False),
            Mod(_critic, ["_critic_input"], ["state_value"])
        ).to(self.device)

        if self.cfg.train_dr_estimator:
            assert "dr_" in observation_spec, "dr_ should be in observation_spec"
            dr_shape = observation_spec["dr_"].shape[-1]
            mlp = nn.Sequential(
                make_mlp([latent_dim, latent_dim]),
                nn.LazyLinear(dr_shape),
            )
            self.dr_estimator = Mod(mlp, [PRIV_PRED_KEY], ["dr_pred"]).to(self.device)
            
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["adapt_hx"] = torch.zeros(fake_input.shape[0], latent_dim)
            fake_input["depth_hx"] = torch.zeros(fake_input.shape[0], self.depth_feature_dim)
            fake_input["_depth_feature"] = torch.zeros(fake_input.shape[0], self.depth_feature_dim)

        self.object_transform(fake_input)
        if has_height:
            self.height_encoder(fake_input)
        if has_depth:
            # Initialize TemporalDepthGRU with a fake single depth frame [N, 1, H, W]
            depth_spec = observation_spec[DEPTH_KEY]
            fake_depth = torch.zeros(*depth_spec.shape, device=self.device)
            fake_input[DEPTH_KEY] = fake_depth
            self.temporal_depth_gru(fake_input)
        if self.cfg.use_object_adapt:
            self.object_adapt(fake_input)
            self.object_pred_transform(fake_input)
        else:
            # noobj ablation: adapt_module won't use object_pred (not in adapt_module_in_keys)
            pass
        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)
        if self.cfg.train_dr_estimator:
            self.dr_estimator(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.apply(init_)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)
        if self.cfg.use_object_adapt:
            self.object_adapt_ema = copy.deepcopy(self.object_adapt).requires_grad_(False)
        if has_depth:
            self.temporal_depth_gru_ema = copy.deepcopy(self.temporal_depth_gru).requires_grad_(False)

        self.lr_policy = cfg.lr
        if self.cfg.phase == "train":
            policy_params = [
                    {"params": self.actor.parameters()},
                    {"params": self.encoder_priv.parameters()},
                ]
            if has_height:
                # Teacher: include height CNN in policy optimizer
                policy_params.append({"params": self.height_cnn.parameters()})
        else:
            policy_params = [
                    {"params": self.actor_adapt.parameters()},
                ]
            
        self.opt_policy = torch.optim.Adam(
            policy_params,
            lr=self.lr_policy,
        )
        self.opt_critic = torch.optim.Adam(
            [
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
        )

        # opt_adapt: used in train phase (_train_adapt_no_depth) and finetune phase (train_adapt)
        adapt_params = [{"params": self.adapt_module.parameters()}]
        if self.cfg.use_object_adapt:
            adapt_params.append({"params": self.object_adapt.parameters()})
        if has_depth:
            adapt_params.append({"params": self.temporal_depth_gru.parameters()})
        self.opt_adapt = torch.optim.Adam(adapt_params, lr=cfg.lr)
        if cfg.enable_residual_distillation:
            self.opt_adapt_actor = torch.optim.Adam(
                [
                    {"params": self.actor_adapt.parameters()},
                ],
                lr=cfg.lr,
            )
        if self.cfg.train_dr_estimator:
            self.opt_dr_estimator = torch.optim.Adam(
                [
                    {"params": self.dr_estimator.parameters()},
                ],
                lr=cfg.lr,
            )
        self.num_updates = 0
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = Unbounded((num_envs, self.cfg.latent_dim), device=self.device)
        depth_spec = Unbounded((num_envs, self.depth_feature_dim), device=self.device)
        primers = {}
        if self.cfg.adapt_module == "gru":
            primers["adapt_hx"] = spec
        if hasattr(self, "temporal_depth_gru"):
            primers["depth_hx"] = depth_spec
        return TensorDictPrimer(primers, reset_key="done")

    def get_rollout_policy(self, mode: str="train"):
        modules = []
        has_depth = hasattr(self, "temporal_depth_gru")
        has_height = hasattr(self, "height_encoder")

        if self.cfg.phase == "train":
            modules.append(self.object_transform)
            if has_height:
                modules.append(self.height_encoder)
            modules.append(self.encoder_priv)
            modules.append(self.actor)
            if mode == "train":
                # collect priv_pred for logging during training rollout only
                modules.append(ZeroDepthInjector(self.depth_feature_dim, self.device))
                modules.append(self.adapt_module)
        elif self.cfg.phase == "finetune":
            if has_depth:
                modules.append(self.temporal_depth_gru_ema)
            else:
                modules.append(ZeroDepthInjector(self.depth_feature_dim, self.device))
            if self.cfg.use_object_adapt:
                modules.append(self.object_adapt_ema)
                modules.append(self.object_pred_transform)
            modules.append(self.adapt_ema)
            modules.append(self.actor_adapt)

        out_keys = ["sample_log_prob", "action"] + self.dist_keys
        if self.cfg.adapt_module == "gru" and mode == "train":
            out_keys.append(("next", "adapt_hx"))
        if has_depth and mode == "train":
            out_keys.append(("next", "depth_hx"))
        if self.cfg.phase == "finetune":
            out_keys.append(PRIV_PRED_KEY)
            if self.cfg.adapt_module == "gru":
                out_keys.append(("next", "adapt_hx"))
            if has_depth:
                out_keys.append(("next", "depth_hx"))

        if self.cfg.train_dr_estimator:
            modules.append(self.dr_estimator)
            out_keys.append("dr_pred")

        policy = Seq(*modules, selected_out_keys=out_keys)
        return policy
    
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        info = {}
        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            # Pre-train adapt modules with zero depth injection
            info.update(self._train_adapt_no_depth(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        else:
            raise ValueError(f"Invalid phase: {self.cfg.phase}")

        self.num_updates += 1

        actor = self.actor if self.cfg.phase == "train" else self.actor_adapt
        action_std = actor.module[0][2].module.actor_std.detach()
        for joint_name, std in zip(self.joint_names, action_std):
            info[f"actor_std/{joint_name}"] = std
        info["actor_std/mean"] = action_std.mean()
        return info
    
    def train_policy(self, tensordict: TensorDict):    
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        # entropy coef schedule
        current_iter = self.env.current_iter
        entropy_progress = float(np.clip(current_iter / self.cfg.entropy_decay_iters, 0., 1.))
        self.entropy_coef = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * entropy_progress

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = {}
                info.update(self._update_ppo(minibatch))
                infos.append(info)

                if self.desired_kl is not None: # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    if kl > self.desired_kl * 2.0:
                        self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr_policy = min(1e-2, self.lr_policy * 1.5)
        
                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = self.lr_policy
                    
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr_policy
        infos["actor/entropy_coef"] = self.entropy_coef

        ret = tensordict["ret"]
        ret_mean = ret.mean(dim=(0, 1))
        ret_std = ret.std(dim=(0, 1))
        for i, group_name in enumerate(self.reward_groups):
            infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
            infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
            infos[f"critic/{group_name}.neg_rew_ratio"] = (tensordict[REWARD_KEY][:, :, i] <= 0.).float().mean().item()
        return dict(sorted(infos.items()))
    
    @set_recurrent_mode(True)
    def _train_adapt_no_depth(self, tensordict: TensorDict):
        """Train adapt_module + object_adapt without depth (inject zeros).

        Also distills teacher actor → actor_adapt using GT PRIV_FEATURE_KEY
        as the latent input for actor_adapt.
        """
        infos = []
        with torch.no_grad():
            self.object_transform(tensordict)
            if hasattr(self, "height_encoder"):
                self.height_encoder(tensordict)
            self.encoder_priv(tensordict)

        for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
            # Inject zero depth feature (no depth sensor available in train phase)
            minibatch["_depth_feature"] = torch.zeros(
                *minibatch.batch_size, self.depth_feature_dim, device=self.device
            )

            # Supervised loss: object prediction
            if self.cfg.use_object_adapt:
                self.object_adapt(minibatch)
                object_loss = self.adapt_loss_fn(minibatch[OBJECT_PRED_KEY], minibatch[OBJECT_KEY])
                object_loss = (object_loss * (~minibatch["is_init"])).mean()
            else:
                object_loss = torch.zeros(1, device=self.device)

            # Supervised loss: priv feature prediction
            if self.cfg.use_object_adapt:
                self.object_pred_transform(minibatch)
            self.adapt_module(minibatch)
            priv_loss = self.adapt_loss_fn(minibatch[PRIV_PRED_KEY], minibatch[PRIV_FEATURE_KEY])
            priv_loss = (priv_loss * (~minibatch["is_init"])).mean()

            total_sup = priv_loss + object_loss
            self.opt_adapt.zero_grad()
            total_sup.backward()
            all_params = list(self.adapt_module.parameters())
            if self.cfg.use_object_adapt:
                all_params += list(self.object_adapt.parameters())
            if hasattr(self, "temporal_depth_gru"):
                all_params += list(self.temporal_depth_gru.parameters())
            opt_adapt_grad_norm = nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)
            self.opt_adapt.step()

            info = {}
            info["adapt/priv_loss"] = priv_loss.detach()
            info["adapt/object_loss"] = object_loss.detach()
            info["adapt/grad_norm"] = opt_adapt_grad_norm
            info["adapt/priv_feature_norm"] = minibatch[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean()
            info["adapt/priv_pred_norm"] = minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean()
            if "_depth_feature" in minibatch.keys():
                info["adapt/depth_feature_norm"] = minibatch["_depth_feature"].norm(p=2, dim=-1).mean()
            if "_height_feature" in minibatch.keys():
                info["adapt/height_feature_norm"] = minibatch["_height_feature"].norm(p=2, dim=-1).mean()

            # Actor distillation using GT PRIV_FEATURE_KEY
            if self.cfg.enable_residual_distillation:
                with torch.no_grad():
                    dist_teacher = self.actor.get_dist(minibatch)
                minibatch[PRIV_PRED_KEY] = minibatch[PRIV_FEATURE_KEY].detach()
                dist_student = self.actor_adapt.get_dist(minibatch)
                adapt_loss = (dist_teacher.mean - dist_student.mean).square().mean()

                self.opt_adapt_actor.zero_grad()
                adapt_loss.backward()
                self.opt_adapt_actor.step()
                info["adapt/adapt_loss"] = adapt_loss.detach()

            infos.append(TensorDict(info, []))

        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    @set_recurrent_mode(True)
    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.object_transform(tensordict)
            if hasattr(self, "height_encoder"):
                # Encode height map to get teacher priv_feature targets for distillation
                self.height_encoder(tensordict)
            self.encoder_priv(tensordict)

        for epoch in range(2):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                # Student: encode depth image sequence via TemporalDepthGRU
                if hasattr(self, "temporal_depth_gru"):
                    self.temporal_depth_gru(minibatch)
                else:
                    # No depth sensor, inject zero _depth_feature
                    minibatch["_depth_feature"] = torch.zeros(
                        *minibatch.batch_size, self.depth_feature_dim, device=self.device
                    )
                if self.cfg.use_object_adapt:
                    self.object_adapt(minibatch)
                    object_loss = self.adapt_loss_fn(minibatch[OBJECT_PRED_KEY], minibatch[OBJECT_KEY])
                    object_loss = (object_loss * (~minibatch["is_init"])).mean()
                else:
                    object_loss = torch.zeros(1, device=self.device)

                if self.cfg.use_object_adapt:
                    self.object_pred_transform(minibatch)
                self.adapt_module(minibatch)
                priv_loss = self.adapt_loss_fn(minibatch[PRIV_PRED_KEY], minibatch[PRIV_FEATURE_KEY])
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()

                total_loss = priv_loss + object_loss

                self.opt_adapt.zero_grad()
                total_loss.backward()

                all_params = list(self.adapt_module.parameters())
                if self.cfg.use_object_adapt:
                    all_params += list(self.object_adapt.parameters())
                if hasattr(self, "temporal_depth_gru"):
                    all_params += list(self.temporal_depth_gru.parameters())
                opt_adapt_grad_norm = nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)
                self.opt_adapt.step()

                info = {}
                info["adapt/priv_loss"] = priv_loss
                info["adapt/object_loss"] = object_loss
                info["adapt/grad_norm"] = opt_adapt_grad_norm
                info["adapt/priv_feature_norm"] = minibatch[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean()
                info["adapt/priv_pred_norm"] = minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean()
                if "_depth_feature" in minibatch.keys():
                    info["adapt/depth_feature_norm"] = minibatch["_depth_feature"].norm(p=2, dim=-1).mean()
                if "_height_feature" in minibatch.keys():
                    info["adapt/height_feature_norm"] = minibatch["_height_feature"].norm(p=2, dim=-1).mean()

                if self.cfg.enable_residual_distillation:
                    # Distillation in finetune phase: student actor_adapt mimics teacher actor output.
                    with torch.no_grad():
                        dist_teacher = self.actor.get_dist(minibatch)

                    # depth_encoder only updates via supervised loss (priv_loss/object_loss),
                    # NOT via actor distillation gradients — detach _depth_feature here.
                    if "_depth_feature" in minibatch.keys():
                        minibatch["_depth_feature"] = minibatch["_depth_feature"].detach()

                    minibatch[PRIV_PRED_KEY] = minibatch[PRIV_PRED_KEY].detach()
                    dist_student = self.actor_adapt.get_dist(minibatch)

                    adapt_loss = (dist_teacher.mean - dist_student.mean).square().mean()

                    self.opt_adapt_actor.zero_grad()
                    adapt_loss.backward()
                    self.opt_adapt_actor.step()
                    info["adapt/adapt_loss"] = adapt_loss
                
                if self.cfg.train_dr_estimator:
                    minibatch[PRIV_PRED_KEY] = minibatch[PRIV_PRED_KEY].detach()
                    self.dr_estimator(minibatch)
                    
                    dr_est_loss = (minibatch["dr_pred"] - minibatch["dr_"]).square().mean()
                    self.opt_dr_estimator.zero_grad()
                    dr_est_grad_norm = nn.utils.clip_grad_norm_(self.dr_estimator.parameters(), self.cfg.max_grad_norm)
                    dr_est_loss.backward()
                    self.opt_dr_estimator.step()
                    info["adapt/dr_est_grad_norm"] = dr_est_grad_norm
                    info["adapt/dr_est_loss"] = dr_est_loss
                    
                infos.append(TensorDict(info, []))
        
        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        if self.cfg.use_object_adapt:
            soft_copy_(self.object_adapt, self.object_adapt_ema, 0.04)
        if hasattr(self, "temporal_depth_gru"):
            soft_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema, 0.04)
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos
    
    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        # with tensordict.view(-1) as tensordict_flat:
        #     critic(tensordict_flat)
        #     critic(tensordict_flat["next"])
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        # Compute and normalize the advantages
        # [num_steps, num_envs, num_reward_groups]
        if self.cfg.normalize_before_sum: # normalize, scale, sum
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 0.01)
            adv_norm *= self.reward_scales
            # [num_steps, num_envs, num_reward_groups]
            adv_norm_sum = adv_norm.sum(dim=2, keepdim=True)
            # [num_steps, num_envs, 1]
            adv_final = adv_norm_sum
        else: # scale, sum, normalize
            adv *= self.reward_scales
            adv_sum = adv.sum(dim=2, keepdim=True)
            # [num_steps, num_envs, 1]
            adv_sum_norm = (adv_sum - adv_sum.mean(dim=(0, 1))) / (adv_sum.std(dim=(0, 1)) + 1e-8)
            # [num_steps, num_envs, 1]
            adv_final = adv_sum_norm

        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv_final)
        # shape: (N, T, 1)
        tensordict.set(ret_key, ret)
        tensordict["adv_before_norm"] = adv
        # shape: (N, T, num_reward_groups)
        return tensordict

    # @torch.compile
    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)

        if self.cfg.phase == "train":
            # Teacher update: encode height map, run encoder_priv, compute actor loss
            self.object_transform(tensordict)
            if hasattr(self, "height_encoder"):
                self.height_encoder(tensordict)
            self.encoder_priv(tensordict)
            actor = self.actor
        elif self.cfg.phase == "finetune":
            if hasattr(self, "temporal_depth_gru_ema"):
                self.temporal_depth_gru_ema(tensordict)
            else:
                # No depth sensor, inject zero _depth_feature
                tensordict["_depth_feature"] = torch.zeros(
                    *tensordict.batch_size, self.depth_feature_dim, device=self.device
                )
            if self.cfg.use_object_adapt:
                self.object_adapt_ema(tensordict)
                self.object_pred_transform(tensordict)
            self.adapt_ema(tensordict)
            actor = self.actor_adapt
        else:
            raise ValueError(f"Invalid phase: {self.cfg.phase}")

        dist: D.Independent = actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        if self.cfg.phase == "train":
            valid = (tensordict["step_count"] > 1)
        else:
            valid = (tensordict["step_count"] > 5)
        valid = valid.squeeze(-1)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        if self.cfg.normalize_ratio:
            clamped_ratio = ratio.clamp(1.-self.clip_param, 1.+self.clip_param).detach()
            surr1 = surr1 / clamped_ratio
            surr2 = surr2 / clamped_ratio
        policy_loss = - (torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = value_loss[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()

        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        if self.cfg.phase == "train":
            priv_grad_norm = nn.utils.clip_grad_norm_(self.encoder_priv.parameters(), self.cfg.max_grad_norm)
            if hasattr(self, "height_cnn"):
                nn.utils.clip_grad_norm_(self.height_cnn.parameters(), self.cfg.max_grad_norm)
        else:
            priv_grad_norm = torch.zeros(1)
        self.opt_policy.step()
        self.opt_critic.step()
        
        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            # loc, scale = dist.loc, dist.scale
            # kl = torch.sum(
            #     torch.log(scale) - torch.log(scale_old)
            #     + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
            #     - 0.5,
            #     dim=-1,
            # ).mean()
            dist_old = self.dist_cls(**dist_kwargs_old)
            kl = D.kl_divergence(dist_old, dist).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/priv_grad_norm": priv_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    def state_dict(self):
        if self.cfg.phase == "train":
            if not self.cfg.enable_residual_distillation:
                hard_copy_(self.actor, self.actor_adapt)
            else:
                actor_std = self.actor.module[0][2].module.actor_std
                actor_adapt_std = self.actor_adapt.module[0][2].module.actor_std
                actor_adapt_std.data.copy_(actor_std.data)
            
        if self.cfg.phase == "train":
            hard_copy_(self.adapt_module, self.adapt_ema)
            if self.cfg.use_object_adapt:
                hard_copy_(self.object_adapt, self.object_adapt_ema)
            if hasattr(self, "temporal_depth_gru"):
                hard_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema)
        
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
        state_dict["last_iter"] = self.env.current_iter
        state_dict["lr_policy"] = self.lr_policy
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        self.env.set_progress(state_dict.get("last_iter", 0))
        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy

        return failed_keys
