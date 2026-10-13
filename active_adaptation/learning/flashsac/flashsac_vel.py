"""VAIC teacher actor/adaptation with a FlashSAC actor and flat FlashSAC critic.

The actor sees the same semantic inputs as ``ppo_vel_train``: command, policy
observation, and a learned privileged feature.  Replay stores the raw inputs to
that feature encoder, so every sampled actor and bootstrap observation is
encoded with the current weights.  The fixed object point template is kept once
in the actor and transformed from each replayed object pose instead of storing
``object_trans`` in every row.  The critic keeps the observation layout and
architecture from ``flashsac_vel_flat_train``.

The GRU adaptation module and ``actor_adapt`` are auxiliary student modules.
They are trained from the latest contiguous rollout only; recurrent hidden
states are never put in the long-term SAC replay.
"""

import copy
import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase
from tensordict.nn import TensorDictSequential as Seq
from torchrl.data import Unbounded
from torchrl.envs.transforms import TensorDictPrimer
from termcolor import colored

from ..modules.rnn import set_recurrent_mode
from ..ppo.common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY, CatTensors, make_batch, make_conv, make_mlp
from .flashsac_upstream.agent import _sample_flashsac_actions, _update_networks
from .flashsac_upstream.network import FlashSACActor
from .flashsac_upstream.scheduler import ramp_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .flashsac_vel_flat import (
    ACTION_DR_KEY,
    U_KEY,
    FlashSACVelFlat,
    FlashSACVelFlatConfig,
    NStepTrajectoryReplay,
    _raw,
)
from ..ppo.ppo_vel import (
    HEIGHT_KEY,
    OBJECT_GEO_KEY,
    OBJECT_KEY,
    OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY,
    OBJECT_TRANS_KEY,
    PRIV_FEATURE_KEY,
    PRIV_PRED_KEY,
    REF_JPOS_KEY,
    VEL_CMD_KEY,
    GRUModule,
    TransformObject,
)


ACTOR_REPLAY_KEY = "_flashsac_actor_replay"
DEPTH_FEATURE_KEY = "_depth_feature"


@dataclass
class FlashSACVelConfig(FlashSACVelFlatConfig):
    _target_: str = "active_adaptation.learning.flashsac.flashsac_vel.FlashSACVel"
    name: str = "flashsac_vel"

    # PPOVEL-compatible privileged/adaptation architecture.
    latent_dim: int = 256
    adapt_module: str = "gru"
    adapt_module_input_cmd: bool = True
    use_object_adapt: bool = True
    enable_residual_distillation: bool = True
    adapt_learning_rate: float = 3e-4
    adapt_num_minibatches: int = 8
    max_grad_norm: float = 1.0

    # Full actor command/policy plus the critic-selected privileged inputs produce a 1,580-dim
    # replay row on the skateboard task. Keep the long replay in host RAM by default and stage
    # sampled minibatches to the GPU through pinned memory.
    buffer_max_length: int = 10_000_000
    buffer_device_type: str = "cpu"


cs = ConfigStore.instance()
cs.store("flashsac_vel_train", node=FlashSACVelConfig, group="algo")


class VAICFlashSACActor(nn.Module):
    """FlashSAC policy preceded by PPOVEL's height and privileged encoders."""

    points_per_object = 128
    object_transform_dim = 12

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        replay_layout,
        command_key: str,
        latent_dim: int,
        object_points_template: torch.Tensor | None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.command_key = command_key
        self.layout = {key: (start, stop, tuple(shape)) for key, start, stop, shape in replay_layout}

        required = [command_key, OBS_KEY, OBS_PRIV_KEY]
        missing = [key for key in required if key not in self.layout]
        if missing:
            raise ValueError(f"VAIC FlashSAC actor is missing replay fields {missing}")

        self.has_object = OBJECT_KEY in self.layout
        if self.has_object and object_points_template is None:
            raise ValueError("VAIC FlashSAC actor needs a fixed object point template")
        if object_points_template is not None:
            object_points_template = object_points_template.reshape(-1, 3)
            if object_points_template.shape[-2] % self.points_per_object:
                raise ValueError(
                    f"object template has {object_points_template.shape[-2]} points; "
                    f"expected a multiple of {self.points_per_object}"
                )
        self.register_buffer("object_points_template", object_points_template)
        self.has_height = HEIGHT_KEY in self.layout
        if self.has_height:
            self.height_cnn = nn.Sequential(
                make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
                nn.LazyLinear(64),
                nn.LayerNorm(64),
            )

        priv_dim = latent_dim
        self.encoder_priv = nn.Sequential(make_mlp([priv_dim]), nn.LazyLinear(priv_dim))
        actor_input_dim = self._dim(command_key) + self._dim(OBS_KEY) + latent_dim
        self.policy = FlashSACActor(num_blocks, actor_input_dim, hidden_dim, action_dim)

        # Materialize PPO-style lazy layers before the optimizer is constructed.
        with torch.no_grad():
            self.encode_priv(torch.zeros(2, input_dim))

        def init_encoder(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.encoder_priv.apply(init_encoder)
        if self.has_height:
            self.height_cnn.apply(init_encoder)

    def _dim(self, key: str) -> int:
        start, stop, _ = self.layout[key]
        return stop - start

    def field(self, observation: torch.Tensor, key: str) -> torch.Tensor:
        start, stop, shape = self.layout[key]
        value = observation[..., start:stop]
        return value.reshape(*observation.shape[:-1], *shape)

    def transform_object_pose(self, object_vec: torch.Tensor) -> torch.Tensor:
        """Transform the fixed object point template with replayed object poses."""
        if self.object_points_template is None:
            raise RuntimeError("object point template is unavailable")
        points = self.object_points_template.to(dtype=object_vec.dtype)
        objects_num = points.shape[-2] // self.points_per_object
        transform_dim = self.object_transform_dim
        required_dim = objects_num * transform_dim
        if object_vec.shape[-1] < required_dim:
            raise ValueError(
                f"object pose has {object_vec.shape[-1]} values; at least {required_dim} are required"
            )
        object_vec = object_vec[..., -required_dim:]
        transformed = []
        for i in range(objects_num):
            start = i * transform_dim
            pos = object_vec[..., start:start + 3]
            ori = object_vec[..., start + 3:start + 12].reshape(*object_vec.shape[:-1], 3, 3)
            object_points = points[
                i * self.points_per_object:(i + 1) * self.points_per_object
            ]
            transformed.append(torch.matmul(object_points, ori.transpose(-1, -2)) + pos.unsqueeze(-2))
        return torch.cat(transformed, dim=-2).flatten(-2, -1)

    def encode_priv(self, observation: torch.Tensor, *, detach_inputs: bool = False) -> torch.Tensor:
        inputs = [self.field(observation, OBS_PRIV_KEY)]
        if self.has_object:
            inputs.extend([
                self.field(observation, OBJECT_KEY),
                self.transform_object_pose(self.field(observation, OBJECT_KEY)),
            ])
        if self.has_height:
            inputs.append(self.height_cnn(self.field(observation, HEIGHT_KEY)))
        inputs = torch.cat(inputs, dim=-1)
        # Auxiliary teacher alignment must not train upstream feature encoders.
        return self.encoder_priv(inputs.detach() if detach_inputs else inputs)

    def actor_input(self, observation: torch.Tensor) -> torch.Tensor:
        priv_feature = self.encode_priv(observation)
        return torch.cat([
            self.field(observation, self.command_key),
            self.field(observation, OBS_KEY),
            priv_feature,
        ], dim=-1)

    def get_mean_and_std(self, observations: torch.Tensor, training: bool):
        return self.policy.get_mean_and_std(self.actor_input(observations), training)

    def forward(self, observations: torch.Tensor, training: bool):
        return self.policy(self.actor_input(observations), training)


class FlashSACVelRollout(TensorDictModuleBase):
    def __init__(self, policy: "FlashSACVel", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.deterministic = mode != "train"
        self.train_mode = mode == "train"
        self.in_keys = list(policy.replay_env_keys)
        self.out_keys = [ACTION_KEY, U_KEY]
        if self.train_mode:
            if VEL_CMD_KEY not in self.in_keys:
                self.in_keys.append(VEL_CMD_KEY)
            self.in_keys.append("is_init")
            self.out_keys.append(PRIV_PRED_KEY)
            if policy.cfg.adapt_module == "gru":
                self.in_keys.append("adapt_hx")
                self.out_keys.append(("next", "adapt_hx"))

    def forward(self, tensordict: TensorDictBase):
        policy = self.policy
        if ACTION_DR_KEY in policy.replay_keys:
            tensordict.set(ACTION_DR_KEY, policy.action_dr())
        actor_observation = policy.replay_obs(tensordict)
        u = policy.act(actor_observation, self.deterministic)
        tensordict.set(U_KEY, u)
        if policy.cfg.action_mode == "residual":
            action = tensordict[REF_JPOS_KEY] + policy.cfg.residual_scale * u
        else:
            action = policy.cfg.action_scale * u
        tensordict.set(ACTION_KEY, action)

        if self.train_mode:
            policy.set_object_trans(tensordict)
            tensordict[DEPTH_FEATURE_KEY] = torch.zeros(
                *tensordict.batch_size, policy.depth_feature_dim, device=policy.device
            )
            policy.adapt_module(tensordict)
        return tensordict


class FlashSACVel(FlashSACVelFlat):
    """FlashSAC teacher with PPOVEL actor inputs and auxiliary adaptation."""

    def __init__(self, cfg: FlashSACVelConfig, observation_spec, action_spec, reward_spec, device, env):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self.cmd_key = "command_" if observation_spec.get("command_", None) is not None else CMD_KEY

        # Replay keeps full command/policy for the PPO-style actor, but stores only the privileged
        # entries selected by the proven flat FlashSAC critic.  Other critic-only groups are also
        # stored compactly.  This preserves the actor's direct tracking/proprioceptive inputs while
        # avoiding the three unused future steps in the large privileged group.
        self.replay_keys = list(self.obs_keys)
        actor_keys = [self.cmd_key, OBS_KEY, OBS_PRIV_KEY]
        if observation_spec.get(OBJECT_KEY, None) is not None:
            actor_keys.append(OBJECT_KEY)
        if observation_spec.get(HEIGHT_KEY, None) is not None:
            actor_keys.append(HEIGHT_KEY)
        for key in actor_keys:
            if key not in self.replay_keys:
                self.replay_keys.append(key)

        critic_field_index = self._critic_field_indices(observation_spec)
        actor_full_keys = {self.cmd_key, OBS_KEY, OBJECT_KEY, HEIGHT_KEY}
        replay_layout = []
        self._replay_field_index = {}
        offset = 0
        for key in self.replay_keys:
            full_shape = self._replay_field_shape(key, observation_spec)
            full_dim = math.prod(full_shape)
            selected = critic_field_index.get(key)
            if key == OBS_PRIV_KEY:
                if selected is None:
                    raise ValueError("flashsac_vel_train requires priv in the critic observation")
                field_index = selected
            elif key in actor_full_keys or key not in self.obs_keys:
                field_index = torch.arange(full_dim, device=self.device)
            else:
                field_index = selected
                if field_index is None:
                    field_index = torch.arange(full_dim, device=self.device)
            is_full = field_index.numel() == full_dim and torch.equal(
                field_index, torch.arange(full_dim, device=self.device)
            )
            self._replay_field_index[key] = None if is_full else field_index
            shape = full_shape if is_full else (field_index.numel(),)
            dim = math.prod(shape)
            replay_layout.append((key, offset, offset + dim, shape))
            offset += dim
        self.replay_layout = replay_layout
        self.replay_obs_dim = offset
        self._critic_replay_index = self._make_critic_replay_index(critic_field_index)
        if self._critic_replay_index.numel() != self.obs_dim:
            raise RuntimeError(
                f"critic replay index has {self._critic_replay_index.numel()} entries, "
                f"expected {self.obs_dim}"
            )
        self.replay_env_keys = []
        for key in self.replay_keys:
            if key == OBJECT_TRANS_KEY:
                self.replay_env_keys.append(OBJECT_KEY)
            elif key != ACTION_DR_KEY:
                self.replay_env_keys.append(key)
        self.replay_env_keys = list(dict.fromkeys(self.replay_env_keys))

        object_points_template = self._fixed_object_points(env, observation_spec)

        # Replace only the parent's flat actor.  The critic, target critic, temperature, reward
        # normalization, exploration process, and update rule remain the FlashSAC implementation.
        self._actor = self._init_vaic_actor(replay_layout, object_points_template)
        self.actor = _raw(self._actor)

        self.depth_feature_dim = 64
        self._build_adaptation(observation_spec)
        self._adapt_steps = []
        self._pending_adapt = None

        print(colored(
            f"[FlashSAC VAIC] replay observation {self.replay_obs_dim} dims; "
            f"critic selects {self.obs_dim}, actor recomputes a {cfg.latent_dim}-dim privileged feature.",
            "green",
        ))

    def _replay_field_shape(self, key, observation_spec):
        if key == ACTION_DR_KEY:
            return (2,)
        if key == OBJECT_TRANS_KEY:
            return (math.prod(observation_spec[OBJECT_GEO_KEY].shape[1:]),)
        return tuple(observation_spec[key].shape[1:])

    def _critic_field_indices(self, observation_spec):
        """Convert the parent's flat critic selection into per-observation-group indices."""
        selected_global = self._obs_index.tolist()
        result = {}
        full_offset = 0
        cursor = 0
        for key in self.obs_keys:
            full_dim = math.prod(self._replay_field_shape(key, observation_spec))
            local = []
            while cursor < len(selected_global) and selected_global[cursor] < full_offset + full_dim:
                local.append(selected_global[cursor] - full_offset)
                cursor += 1
            result[key] = torch.tensor(local, dtype=torch.long, device=self.device)
            full_offset += full_dim
        if cursor != len(selected_global):
            raise RuntimeError("failed to partition the critic observation selection")
        return result

    def _make_critic_replay_index(self, critic_field_index):
        layout = {key: (start, stop) for key, start, stop, _ in self.replay_layout}
        result = []
        for key in self.obs_keys:
            start, stop = layout[key]
            replay_index = self._replay_field_index[key]
            selected = critic_field_index[key]
            if replay_index is None:
                result.extend(start + original for original in selected.tolist())
                continue
            positions = {original.item(): i for i, original in enumerate(replay_index)}
            try:
                result.extend(start + positions[original.item()] for original in selected)
            except KeyError as error:
                raise RuntimeError(f"replay field {key} omits a critic entry") from error
            if stop - start != replay_index.numel():
                raise RuntimeError(f"invalid compact replay layout for {key}")
        return torch.as_tensor(result, dtype=torch.long, device=self.device)

    def _fixed_object_points(self, env, observation_spec):
        if observation_spec.get(OBJECT_KEY, None) is None:
            return None
        if observation_spec.get(OBJECT_GEO_KEY, None) is None:
            raise ValueError("object_ is present but object_geo_ is missing")
        with torch.no_grad():
            geometry = env.observation_funcs[OBJECT_GEO_KEY]._compute().reshape(self.num_envs, -1, 3)
        template = geometry[0].detach()
        if not torch.allclose(geometry, template.unsqueeze(0).expand_as(geometry)):
            raise ValueError(
                "flashsac_vel_train can omit object_trans from replay only when object_geo_ "
                "is identical in every environment"
            )
        return template.cpu().clone()

    def _init_vaic_actor(self, replay_layout, object_points_template):
        cfg = self.cfg
        actor_net = VAICFlashSACActor(
            num_blocks=cfg.actor_num_blocks,
            input_dim=self.replay_obs_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=self.action_dim,
            replay_layout=replay_layout,
            command_key=self.cmd_key,
            latent_dim=cfg.latent_dim,
            object_points_template=object_points_template,
        ).to(self.device)
        use_fused = self.device.type == "cuda" and torch.cuda.is_available()
        optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
        num_interaction_steps = max(1, cfg.num_env_steps // self.num_envs)
        num_updates = num_interaction_steps * cfg.updates_per_interaction_step
        schedule = ramp_cosine_decay_scheduler(
            init_value=cfg.learning_rate_init,
            peak_value=cfg.learning_rate_peak,
            end_value=cfg.learning_rate_end,
            ramp_steps=int(cfg.learning_rate_ramp_rate * num_updates),
            decay_steps=int(cfg.learning_rate_decay_rate * num_updates),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: schedule(step) / cfg.learning_rate_peak
        )
        actor = Network(
            network=actor_net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile_network=cfg.use_compile,
            compile_mode=self.flash_cfg.compile_mode,
            use_weight_normalization=True,
        )
        if cfg.use_compile:
            actor.network.get_mean_and_std = torch.compile(
                actor.network.get_mean_and_std, mode=self.flash_cfg.compile_mode
            )
        actor.normalize_parameters()
        return actor

    def _build_adaptation(self, observation_spec):
        cfg = self.cfg
        latent_dim = cfg.latent_dim
        object_dim = observation_spec[OBJECT_KEY].shape[-1]
        adapt_in_keys: List[str] = [OBS_KEY]
        if cfg.adapt_module_input_cmd:
            adapt_in_keys.append(VEL_CMD_KEY)
        if cfg.use_object_adapt:
            adapt_in_keys.extend([OBJECT_KEY, OBJECT_TRANS_KEY])

        self.object_pred_transform = Seq(
            TransformObject(object_dim, [OBJECT_PRED_KEY, OBJECT_GEO_KEY], [OBJECT_PRED_TRANS_KEY])
        ).to(self.device)
        if cfg.use_object_adapt:
            self.object_adapt = Seq(
                CatTensors([OBS_KEY, VEL_CMD_KEY], "_object_adapt_mlp_inp", del_keys=False, sort=False),
                Mod(make_mlp([latent_dim]), "_object_adapt_mlp_inp", "_obj_adapt_mlp"),
                CatTensors(["_obj_adapt_mlp", DEPTH_FEATURE_KEY], "_object_adapt_inp", del_keys=False),
                Mod(
                    nn.Sequential(make_mlp([latent_dim, latent_dim]), nn.LazyLinear(object_dim)),
                    "_object_adapt_inp",
                    OBJECT_PRED_KEY,
                ),
                selected_out_keys=[OBJECT_PRED_KEY],
            ).to(self.device)

        if cfg.adapt_module == "gru":
            self.adapt_module = Seq(
                CatTensors(adapt_in_keys, "_adapt_inp", del_keys=False, sort=False),
                Mod(
                    GRUModule(latent_dim),
                    ["_adapt_inp", "is_init", "adapt_hx"],
                    [PRIV_PRED_KEY, ("next", "adapt_hx")],
                ),
                selected_out_keys=[PRIV_PRED_KEY, ("next", "adapt_hx")],
            ).to(self.device)
        elif cfg.adapt_module == "mlp":
            self.adapt_module = Seq(
                CatTensors(adapt_in_keys, "_adapt_inp", del_keys=False, sort=False),
                Mod(
                    nn.Sequential(make_mlp([latent_dim, latent_dim]), nn.LazyLinear(latent_dim)),
                    "_adapt_inp",
                    PRIV_PRED_KEY,
                ),
                selected_out_keys=[PRIV_PRED_KEY],
            ).to(self.device)
        else:
            raise ValueError(f"Invalid adapt module: {cfg.adapt_module}")

        # Keep the student actor architecture identical to the FlashSAC teacher.  Only its
        # semantic input differs: the compact velocity command and GRU-predicted privileged
        # latent replace the teacher's full motion command and encoder_priv latent.
        vel_command_dim = observation_spec[VEL_CMD_KEY].shape[-1]
        policy_dim = observation_spec[OBS_KEY].shape[-1]
        actor_adapt_input_dim = vel_command_dim + policy_dim + latent_dim
        self.actor_adapt = FlashSACActor(
            num_blocks=cfg.actor_num_blocks,
            input_dim=actor_adapt_input_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        self._normalize_actor_adapt_parameters()
        print(colored(
            f"[FlashSAC VAIC] actor_adapt input {actor_adapt_input_dim} dims "
            f"(vel_command {vel_command_dim} + policy {policy_dim} + latent {latent_dim}).",
            "green",
        ))

        fake = observation_spec.zero().to(self.device)
        fake["is_init"] = torch.ones(*fake.batch_size, 1, dtype=torch.bool, device=self.device)
        fake["adapt_hx"] = torch.zeros(*fake.batch_size, latent_dim, device=self.device)
        fake[DEPTH_FEATURE_KEY] = torch.zeros(*fake.batch_size, self.depth_feature_dim, device=self.device)
        self.set_object_trans(fake)
        if cfg.use_object_adapt:
            self.object_adapt(fake)
        self.adapt_module(fake)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.adapt_module.apply(init_)
        adapt_params = list(self.adapt_module.parameters())
        if cfg.use_object_adapt:
            self.object_adapt.apply(init_)
            adapt_params += list(self.object_adapt.parameters())
        self.opt_adapt = optim.Adam(adapt_params, lr=cfg.adapt_learning_rate)
        self.opt_adapt_actor = optim.Adam(self.actor_adapt.parameters(), lr=cfg.adapt_learning_rate)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)
        if cfg.use_object_adapt:
            self.object_adapt_ema = copy.deepcopy(self.object_adapt).requires_grad_(False)

    @torch.no_grad()
    def _normalize_actor_adapt_parameters(self):
        """Apply the same UnitLinear/UnitNorm constraints as the teacher actor."""
        for module in self.actor_adapt.modules():
            normalize = getattr(module, "normalize_parameters", None)
            if normalize is not None:
                normalize()

    def _actor_adapt_input(self, tensordict: TensorDictBase) -> torch.Tensor:
        return torch.cat(
            [tensordict[VEL_CMD_KEY], tensordict[OBS_KEY], tensordict[PRIV_PRED_KEY]],
            dim=-1,
        )

    def make_tensordict_primer(self):
        if self.cfg.adapt_module != "gru":
            return TensorDictPrimer({}, reset_key="done")
        spec = Unbounded((self.num_envs, self.cfg.latent_dim), device=self.device)
        return TensorDictPrimer({"adapt_hx": spec}, reset_key="done")

    def get_rollout_policy(self, mode: str = "train"):
        return FlashSACVelRollout(self, mode)

    def replay_obs(self, tensordict: TensorDictBase) -> torch.Tensor:
        if OBJECT_TRANS_KEY in self.replay_keys and tensordict.get(OBJECT_TRANS_KEY, None) is None:
            self.set_object_trans(tensordict)
        batch_size = tensordict.batch_size
        fields = []
        for key in self.replay_keys:
            value = tensordict[key].reshape(*batch_size, -1).float()
            field_index = self._replay_field_index[key]
            if field_index is not None:
                value = value.index_select(-1, field_index)
            fields.append(value)
        return torch.cat(fields, dim=-1)

    def set_object_trans(self, tensordict: TensorDictBase):
        if OBJECT_KEY in tensordict.keys():
            tensordict[OBJECT_TRANS_KEY] = self.actor.transform_object_pose(tensordict[OBJECT_KEY])
        return tensordict

    def critic_obs(self, replay_observation: torch.Tensor) -> torch.Tensor:
        return replay_observation.index_select(-1, self._critic_replay_index)

    @torch.no_grad()
    def act(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            mean, _ = self.actor.get_mean_and_std(observation, training=False)
            return torch.tanh(mean)
        if not self.can_start_training():
            return torch.rand(observation.shape[0], self.action_dim, device=observation.device) * 2 - 1
        (
            self._cached_noise,
            actions,
            self._cur_noise_repeat_count,
            self._cur_noise_repeat_n,
        ) = _sample_flashsac_actions(
            actor=self._actor,
            noise=self._cached_noise,
            observations=observation,
            temperature=1.0,
            cur_count=self._cur_noise_repeat_count,
            cur_n=self._cur_noise_repeat_n,
            zeta_cdf=self._zeta_cdf,
        )
        return actions

    @torch.no_grad()
    def add_transition(self, tensordict: TensorDictBase):
        next_td = tensordict["next"]
        if ACTION_DR_KEY in self.replay_keys:
            next_td.set(ACTION_DR_KEY, tensordict[ACTION_DR_KEY])
        current_replay = self.replay_obs(tensordict)
        next_replay = self.replay_obs(next_td)

        if self.buffer is None:
            storage_device = "cpu" if self.cfg.buffer_device_type == "cpu" else self.device
            if torch.device(storage_device).type == "cuda":
                capacity = max(
                    self.cfg.buffer_max_length // tensordict.shape[0], self.cfg.n_step + 2
                ) * tensordict.shape[0]
                obs_bytes = torch.finfo(getattr(torch, self.cfg.buffer_obs_dtype)).bits // 8
                required = capacity * (
                    self.replay_obs_dim * obs_bytes + self.action_dim * 4 + 15
                )
                free, _ = torch.cuda.mem_get_info(self.device)
                reserve = 4 * 2**30
                if required > free - reserve:
                    fit = max(0, int(capacity * (free - reserve) / required))
                    raise MemoryError(
                        f"VAIC FlashSAC replay needs {required / 2**30:.1f} GiB but only "
                        f"{free / 2**30:.1f} GiB is free; keeping 4 GiB for training requires "
                        f"algo.buffer_max_length about {fit} or less, or set "
                        "algo.buffer_device_type=cpu."
                    )
            self.buffer = NStepTrajectoryReplay(
                num_envs=tensordict.shape[0],
                obs_dim=self.replay_obs_dim,
                action_dim=self.action_dim,
                n_step=self.cfg.n_step,
                gamma=self.cfg.gamma,
                max_length=self.cfg.buffer_max_length,
                min_length=self.cfg.buffer_min_length,
                batch_size=self.cfg.sample_batch_size,
                device=self.device,
                obs_dtype=getattr(torch, self.cfg.buffer_obs_dtype),
                storage_device=storage_device,
            )
            print(colored(
                f"[FlashSAC VAIC] replay buffer: {self.buffer.capacity} rows, "
                f"{self.buffer.nbytes() / 2**30:.2f} GiB on {self.buffer.storage}",
                "green",
            ))

        reward = next_td["reward"].sum(-1)
        terminated = next_td["terminated"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        time_limit = next_td["step_count"].squeeze(-1) >= self.env.max_episode_length
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            cut = cut | (truncated & ~time_limit)
        self.buffer.add(
            obs=current_replay if self.buffer.step == 0 else None,
            next_obs=next_replay,
            action=tensordict[U_KEY],
            reward=reward,
            discount=next_td["discount"].squeeze(-1),
            done=terminated | truncated,
            cut=cut,
            valid=tensordict["step_count"].squeeze(-1) > 1,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward, terminated, truncated)
        self._queue_adaptation_step(tensordict, current_replay)

    def _queue_adaptation_step(self, tensordict, actor_replay):
        keys = [OBS_KEY, VEL_CMD_KEY, OBJECT_KEY, OBJECT_TRANS_KEY, "is_init", REF_JPOS_KEY]
        if self.cfg.adapt_module == "gru":
            keys.append("adapt_hx")
        values = {key: tensordict[key].detach().clone() for key in keys}
        values[ACTOR_REPLAY_KEY] = actor_replay.detach().clone()
        step = TensorDict(values, batch_size=tensordict.batch_size, device=self.device)
        self._adapt_steps.append(step)
        if len(self._adapt_steps) == self.cfg.train_every:
            self._pending_adapt = torch.stack(self._adapt_steps, dim=1)
            self._adapt_steps.clear()

    def update(self):
        if self._pending_adapt is not None:
            info = self._train_adaptation(self._pending_adapt)
            self._pending_adapt = None
            for key, value in info.items():
                value = torch.as_tensor(value, device=self.device).detach().float()
                self._info_sum[key] = self._info_sum.get(key, 0.0) + value
                self._info_cnt[key] = self._info_cnt.get(key, 0) + 1
        super().update()

    def _update_once(self):
        batch = self.buffer.sample()
        actor_observation = batch["observation"]
        actor_next_observation = batch["next_observation"]
        batch["actor_observation"] = actor_observation
        batch["actor_next_observation"] = actor_next_observation
        batch["observation"] = self.critic_obs(actor_observation)
        batch["next_observation"] = self.critic_obs(actor_next_observation)
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        do_actor_update = self._update_step % self.flash_cfg.actor_update_period == 0
        info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self.flash_cfg,
            do_actor_update=do_actor_update,
            device=self.device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1
        return info

    @set_recurrent_mode(True)
    def _train_adaptation(self, tensordict: TensorDict):
        infos = []
        for minibatch in make_batch(
            tensordict, self.cfg.adapt_num_minibatches, self.cfg.train_every
        ):
            batch_shape = minibatch.batch_size
            raw_actor_obs = minibatch[ACTOR_REPLAY_KEY]
            flat_actor_obs = raw_actor_obs.reshape(-1, self.replay_obs_dim)
            with torch.no_grad():
                priv_feature = self.actor.encode_priv(flat_actor_obs).reshape(
                    *batch_shape, self.cfg.latent_dim
                )
                teacher_mean, _ = self.actor.get_mean_and_std(flat_actor_obs, training=False)
                teacher_u = torch.tanh(teacher_mean).reshape(*batch_shape, self.action_dim)
                if self.cfg.action_mode == "residual":
                    teacher_action = minibatch[REF_JPOS_KEY] + self.cfg.residual_scale * teacher_u
                else:
                    teacher_action = self.cfg.action_scale * teacher_u

            minibatch[PRIV_FEATURE_KEY] = priv_feature
            minibatch[DEPTH_FEATURE_KEY] = torch.zeros(
                *batch_shape, self.depth_feature_dim, device=self.device
            )
            if self.cfg.use_object_adapt:
                self.object_adapt(minibatch)
                object_loss = (minibatch[OBJECT_PRED_KEY] - minibatch[OBJECT_KEY]).square()
                object_loss = (object_loss * (~minibatch["is_init"])).mean()
            else:
                object_loss = torch.zeros((), device=self.device)

            self.adapt_module(minibatch)
            priv_loss = (minibatch[PRIV_PRED_KEY] - priv_feature).square()
            priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
            priv_pred_norm = minibatch[PRIV_PRED_KEY].detach().norm(p=2, dim=-1).mean()
            total_loss = priv_loss + object_loss
            self.opt_adapt.zero_grad()
            total_loss.backward()
            adapt_params = list(self.adapt_module.parameters())
            if self.cfg.use_object_adapt:
                adapt_params += list(self.object_adapt.parameters())
            grad_norm = nn.utils.clip_grad_norm_(adapt_params, self.cfg.max_grad_norm)
            self.opt_adapt.step()

            actor_adapt_loss = torch.zeros((), device=self.device)
            if self.cfg.enable_residual_distillation:
                minibatch[PRIV_PRED_KEY] = priv_feature
                student_input = self._actor_adapt_input(minibatch)
                student_input = student_input.reshape(-1, student_input.shape[-1])
                student_mean, _ = self.actor_adapt.get_mean_and_std(
                    student_input, training=True
                )
                student_u = torch.tanh(student_mean).reshape(
                    *batch_shape, self.action_dim
                )
                if self.cfg.action_mode == "residual":
                    student_action = (
                        minibatch[REF_JPOS_KEY]
                        + self.cfg.residual_scale * student_u
                    )
                else:
                    student_action = self.cfg.action_scale * student_u
                # Distill in environment action space.  The student still uses the exact
                # FlashSAC tanh-Gaussian parameterization used by the teacher.
                actor_adapt_loss = (student_action - teacher_action).square().mean()
                self.opt_adapt_actor.zero_grad()
                actor_adapt_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_adapt.parameters(), self.cfg.max_grad_norm)
                self.opt_adapt_actor.step()
                self._normalize_actor_adapt_parameters()

            infos.append({
                "adapt/priv_loss": priv_loss.detach(),
                "adapt/object_loss": object_loss.detach(),
                "adapt/adapt_loss": actor_adapt_loss.detach(),
                "adapt/grad_norm": grad_norm.detach(),
                "adapt/priv_feature_norm": priv_feature.norm(p=2, dim=-1).mean(),
                "adapt/priv_pred_norm": priv_pred_norm,
            })

        with torch.no_grad():
            for target, source in zip(self.adapt_ema.parameters(), self.adapt_module.parameters()):
                target.lerp_(source, 0.04)
            if self.cfg.use_object_adapt:
                for target, source in zip(self.object_adapt_ema.parameters(), self.object_adapt.parameters()):
                    target.lerp_(source, 0.04)
        return {key: torch.stack([info[key] for info in infos]).mean() for key in infos[0]}

    def state_dict(self):
        state = super().state_dict()
        state["replay_layout"] = [
            (key, start, stop, list(shape)) for key, start, stop, shape in self.replay_layout
        ]
        for name in ("adapt_module", "adapt_ema", "actor_adapt"):
            state[name] = getattr(self, name).state_dict()
        state["actor_adapt_arch"] = "flashsac_v1"
        if self.cfg.use_object_adapt:
            state["object_adapt"] = self.object_adapt.state_dict()
            state["object_adapt_ema"] = self.object_adapt_ema.state_dict()
        state["opt_adapt"] = self.opt_adapt.state_dict()
        state["opt_adapt_actor"] = self.opt_adapt_actor.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        saved_layout = state_dict.get("replay_layout")
        current_layout = [(k, s, e, list(shape)) for k, s, e, shape in self.replay_layout]
        if saved_layout is not None and saved_layout != current_layout:
            raise ValueError("checkpoint VAIC actor replay layout does not match the current task/config")
        saved_critic_layout = state_dict.get("obs_layout")
        if saved_critic_layout is not None and saved_critic_layout != self.obs_layout:
            raise ValueError("checkpoint critic observation layout does not match the current config")
        saved_seam = state_dict.get("action_seam")
        if saved_seam is not None and tuple(saved_seam) != self._action_seam():
            raise ValueError(f"checkpoint action seam {tuple(saved_seam)} != {self._action_seam()}")
        for option in ("critic_state_encoder_dim", "critic_action_encoder_dim"):
            if state_dict.get(option) != getattr(self.cfg, option):
                raise ValueError(f"checkpoint {option} does not match the current config")
        uses_critic_encoder = self.cfg.critic_state_encoder_dim or self.cfg.critic_action_encoder_dim
        if uses_critic_encoder and state_dict.get("critic_encoder_arch") != "relu_branches_v1":
            raise ValueError("checkpoint critic encoder architecture does not match the current config")

        for name, network in self._networks().items():
            _raw(network).load_state_dict(state_dict[name], strict=strict)
            if network.optimizer is not None and f"{name}_optimizer" in state_dict:
                network.optimizer.load_state_dict(state_dict[f"{name}_optimizer"])
                network.scheduler.load_state_dict(state_dict[f"{name}_scheduler"])
        for name in ("adapt_module", "adapt_ema"):
            if name in state_dict:
                getattr(self, name).load_state_dict(state_dict[name], strict=strict)
        actor_adapt_compatible = state_dict.get("actor_adapt_arch") == "flashsac_v1"
        if actor_adapt_compatible and "actor_adapt" in state_dict:
            self.actor_adapt.load_state_dict(state_dict["actor_adapt"], strict=strict)
            self._normalize_actor_adapt_parameters()
        elif "actor_adapt" in state_dict:
            print(colored(
                "[FlashSAC VAIC] checkpoint has the legacy PPO-style actor_adapt; "
                "initialized a new FlashSAC actor_adapt instead.",
                "yellow",
            ))
        if self.cfg.use_object_adapt:
            for name in ("object_adapt", "object_adapt_ema"):
                if name in state_dict:
                    getattr(self, name).load_state_dict(state_dict[name], strict=strict)
        if "opt_adapt" in state_dict:
            self.opt_adapt.load_state_dict(state_dict["opt_adapt"])
        if actor_adapt_compatible and "opt_adapt_actor" in state_dict:
            self.opt_adapt_actor.load_state_dict(state_dict["opt_adapt_actor"])

        if self.reward_normalizer is not None and "reward_normalizer" in state_dict:
            rn, saved = self.reward_normalizer, state_dict["reward_normalizer"]
            if saved["G_r"].numel() == self.num_envs:
                rn.G_r = saved["G_r"].to(self.device)
            rn.G_r_max = saved["G_r_max"].to(self.device)
            rn.G_rms.mean = saved["G_rms_mean"].to(self.device)
            rn.G_rms.var = saved["G_rms_var"].to(self.device)
            rn.G_rms.count = saved["G_rms_count"].to(self.device)
        self._update_step = state_dict.get("update_step", 0)
        if "grad_scaler" in state_dict:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])
        self.env.set_progress(state_dict.get("last_iter", 0))
        print(colored(f"[FlashSAC VAIC] loaded checkpoint at update step {self._update_step}.", "green"))
        return []
