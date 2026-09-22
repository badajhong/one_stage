"""GPO joint policy updates with exactly the CTS perception/actor architecture."""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from ..modules.rnn import recurrent_mode, set_recurrent_mode
from ..ppo.common import CatTensors, make_batch, make_mlp
from ..ppo.ppo_vel import PPOVEL
from .losses import adapt_penalty, masked_mean, policy_losses
from .networks import GuiderRollout, LatentCritic, LearnerRollout


class GPOPPOVEL(PPOVEL):
    checkpoint_version = 1
    optimizer_names = ("opt_policy", "opt_critic")

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_config(cfg, observation_spec)
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        # Same construction as CTS: shared velocity-command actor, online
        # depth/object/GRU perception, and predicted (never GT) learner objects.
        self.shared_actor = self.actor_adapt
        del self.actor_adapt, self.actor, self.opt_adapt_actor, self.opt_adapt
        del self.adapt_ema, self.object_adapt_ema, self.temporal_depth_gru_ema
        learner_keys = [{"object_": "object_pred", "object_trans": "object_pred_trans"}.get(k, k)
                        for k in self.adapt_module[0].in_keys]
        self.adapt_module = Seq(
            CatTensors(learner_keys, "_adapt_inp", del_keys=False, sort=False),
            self.adapt_module[1], selected_out_keys=["priv_pred", ("next", "adapt_hx")],
        )
        modules = [self.shared_actor, self.encoder_priv, self.temporal_depth_gru,
                   self.object_adapt, self.adapt_module]
        if hasattr(self, "height_cnn"):
            modules.append(self.height_cnn)
        # ModuleList.parameters deduplicates aliases such as depth_cnn.
        policy_params = list(nn.ModuleList(modules).parameters())
        self.opt_policy = torch.optim.Adam(policy_params, lr=cfg.lr)
        critic_keys = list(self.critic[0].in_keys) + ["gpo_latent"]
        self.critic = LatentCritic(Seq(
            CatTensors(critic_keys, "_gpo_critic_input", del_keys=False),
            Mod(nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(len(self.reward_groups))),
                ["_gpo_critic_input"], ["state_value"]),
        )).to(device)
        fake = observation_spec.zero().to(device)
        fake["gpo_latent"] = torch.zeros(*fake.batch_size, cfg.latent_dim, device=device)
        self.critic(fake)
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, .01)
                nn.init.zeros_(module.bias)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.penalty_coef = cfg.gpo.penalty_init

    @staticmethod
    def _validate_config(cfg, spec):
        if cfg.phase != "train" or not cfg.enable_residual_distillation:
            raise ValueError("GPO requires phase=train and enable_residual_distillation=true for VAIC construction.")
        if cfg.adapt_module != "gru" or not cfg.use_depth or not cfg.use_object_adapt or not cfg.adapt_module_input_cmd:
            raise ValueError("GPO requires the CTS depth/object/GRU velocity-command architecture.")
        if cfg.normalize_before_sum or cfg.normalize_ratio or cfg.train_dr_estimator:
            raise ValueError("GPO requires normalize_before_sum=false, normalize_ratio=false and train_dr_estimator=false.")
        if cfg.load_noise_scale is not None:
            raise ValueError("GPO requires load_noise_scale=null for faithful checkpoint loading.")
        if cfg.train_every < 2 or cfg.num_minibatches < 1 or cfg.ppo_epochs < 1:
            raise ValueError("GPO requires train_every>=2 and positive PPO epochs/minibatches.")
        g = cfg.gpo
        if g.variant not in ("clip", "penalty") or g.eval_policy not in ("guider", "learner"):
            raise ValueError("GPO variant must be clip/penalty and eval_policy must be guider/learner.")
        for name, value in (("inner_clip", g.inner_clip), ("clip_param", cfg.clip_param)):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1.")
        for name, value in (("learner_rl_coef", g.learner_rl_coef), ("penalty_init", g.penalty_init)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name, value in (("target_kl", g.target_kl), ("penalty_max", g.penalty_max),
                            ("lr", cfg.lr), ("max_grad_norm", cfg.max_grad_norm)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(g.penalty_factor) or g.penalty_factor <= 1 or g.penalty_init > g.penalty_max:
            raise ValueError("GPO requires penalty_factor>1 and penalty_init<=penalty_max.")
        for key in ("command", "policy", "priv", "depth", "vel_command", "object_", "object_geo_", "ref_joint_pos_"):
            if key not in spec:
                raise ValueError(f"GPO missing observation {key!r}.")

    def validate_training(self):
        n = self.observation_spec.shape[0]
        if n < self.cfg.num_minibatches or n % self.cfg.num_minibatches:
            raise ValueError("GPO num_envs must be divisible by num_minibatches for complete recurrent sequences.")

    def normalize_latent(self, z):
        return F.normalize(z, p=2, dim=-1, eps=1e-6) if self.cfg.gpo.normalize_latent else z

    def encode_guider(self, td):
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        self.encoder_priv(td)
        td["priv_feature"] = self.normalize_latent(td["priv_feature"])
        td["priv_pred"] = td["priv_feature"]
        td["gpo_latent"] = td["priv_feature"]
        return td

    def encode_learner(self, td):
        if not recurrent_mode():
            for key in ("adapt_hx", "depth_hx"):
                td[key] = td[key] * (~td["is_init"]).to(td[key].dtype)
        self.temporal_depth_gru(td)
        self.object_adapt(td)
        self.object_pred_transform(td)
        self.adapt_module(td)
        td["priv_pred"] = self.normalize_latent(td["priv_pred"])
        td["gpo_latent"] = td["priv_pred"]
        return td

    def get_rollout_policy(self, mode="train"):
        role = self.cfg.gpo.eval_policy if mode == "eval" else mode
        if role in ("learner", "deploy"):
            return LearnerRollout(self)
        if role == "train":
            self.validate_training()
        if role in ("train", "guider"):
            return GuiderRollout(self)
        raise ValueError(f"Unknown GPO mode: {mode}")

    def _update_gpo(self, batch):
        guider_td = self.encode_guider(batch.copy())
        learner_td = self.encode_learner(batch.copy())
        guider = self.shared_actor.get_dist(guider_td)
        learner = self.shared_actor.get_dist(learner_td)
        valid = (batch["step_count"] > 1).squeeze(-1)
        losses = policy_losses(guider, learner, batch["action"], batch["sample_log_prob"],
                               batch["adv"].squeeze(-1), valid, self.clip_param,
                               self.cfg.gpo, self.penalty_coef)
        value_errors = (self.critic(guider_td)["state_value"] - batch["ret"]).square()
        value_loss = masked_mean(value_errors.mean(-1), valid)
        entropy = masked_mean(guider.entropy(), valid)
        total = losses["policy"] + value_loss - self.entropy_coef * entropy
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite GPO loss.")
        old = self.dist_cls(batch["loc"], batch["scale"])
        old_kl = masked_mean(kl_divergence(old, guider), valid).detach()
        self.opt_policy.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        total.backward()
        params = [p for group in self.opt_policy.param_groups for p in group["params"]]
        # Report PPO's module-wise pre-clipping norms without changing GPO's
        # joint policy gradient clipping or its optimizer update.
        def grad_norm(module):
            norms = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
            return torch.stack(norms).norm().item() if norms else 0.

        actor_grad = grad_norm(self.shared_actor)
        priv_grad = grad_norm(self.encoder_priv)
        nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm, error_if_nonfinite=True)
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
        self.opt_policy.step()
        self.opt_critic.step()
        if self.desired_kl is not None:
            kl = old_kl.item()
            if kl > self.desired_kl * 2:
                self.lr_policy = max(1e-5, self.lr_policy / 1.5)
            elif 0 < kl < self.desired_kl / 2:
                self.lr_policy = min(1e-2, self.lr_policy * 1.5)
            for group in self.opt_policy.param_groups:
                group["lr"] = self.lr_policy
        log_ratio = (guider.log_prob(batch["action"]) - batch["sample_log_prob"]).detach()
        info = {
            "actor/policy_loss": losses["guider_ppo"].detach().item(),
            "actor/entropy": entropy.detach().item(),
            "actor/mean_std": guider.scale.detach().mean().item(),
            "actor/grad_norm": actor_grad,
            "actor/priv_grad_norm": priv_grad,
            "actor/clamp_ratio": losses["guider_clip_fraction"].item(),
            "actor/kl": old_kl.item(),
            "actor/approx_kl": masked_mean(log_ratio.clamp(max=60).exp() - 1 - log_ratio, valid).item(),
            "critic/grad_norm": critic_grad.item(),
        }
        for i, name in enumerate(self.reward_groups):
            mse = masked_mean(value_errors[..., i], valid).detach()
            returns = batch["ret"][..., i][valid]
            variance = returns.var() if returns.numel() > 1 else mse.new_zeros(())
            info[f"critic/{name}.value_loss"] = mse.item()
            info[f"critic/{name}.explained_var"] = (1 - mse / variance.clamp_min(1e-8)).item()
        # Alignment KL controls penalty adaptation internally; it is not a new
        # dashboard key and must not be confused with PPO's old/new actor KL.
        return info, losses["alignment_kl"].item()

    @set_recurrent_mode(True)
    def train_op(self, td):
        self.validate_training()
        td = td.exclude("stats")
        # One behavior policy, one advantage normalization over all environments.
        self._compute_advantage(td, self.critic)
        progress = min(1., self.num_updates / max(1, self.cfg.entropy_decay_iters))
        self.entropy_coef = self.cfg.entropy_coef_start + progress * (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
        infos = []
        for _ in range(self.cfg.ppo_epochs):
            distances = []
            # Keep depth and time ordering: learner perception is optimized by
            # both imitation KL and auxiliary RL through recurrent replay.
            for batch in make_batch(td, self.cfg.num_minibatches, self.cfg.train_every):
                info, distance = self._update_gpo(batch)
                infos.append(info)
                distances.append(distance)
            if self.cfg.gpo.variant == "penalty":
                distance = sum(distances) / len(distances)
                self.penalty_coef = adapt_penalty(self.penalty_coef, distance, self.cfg.gpo)
        info = {k: sum(x[k] for x in infos) / len(infos) for k in infos[0]}
        info.update({"actor/lr": self.lr_policy, "actor/entropy_coef": self.entropy_coef})
        for i, name in enumerate(self.reward_groups):
            returns = td["ret"][..., i]
            info[f"critic/{name}.ret_mean"] = returns.mean().item()
            info[f"critic/{name}.ret_std"] = returns.std().item()
            info[f"critic/{name}.neg_rew_ratio"] = (td["next", "reward"][..., i] <= 0).float().mean().item()
        action_std = self.shared_actor.module[0][2].module.actor_std.detach()
        info["actor_std/mean"] = action_std.mean().item()
        for name, value in zip(self.joint_names, action_std.tolist()):
            info[f"actor_std/{name}"] = value
        self.num_updates += 1
        return info

    def state_dict(self):
        state = OrderedDict((name, module.state_dict()) for name, module in self.named_children())
        state.update(gpo_version=self.checkpoint_version, last_phase="gpo", last_iter=self.num_updates,
                     num_updates=self.num_updates, lr_policy=self.lr_policy, penalty_coef=self.penalty_coef,
                     variant=self.cfg.gpo.variant, normalize_latent=self.cfg.gpo.normalize_latent)
        state["optimizers"] = {name: getattr(self, name).state_dict() for name in self.optimizer_names}
        return state

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get("gpo_version") != self.checkpoint_version:
            raise ValueError("Expected a GPO checkpoint; CTS/SITT/VAIC checkpoints are not GPO resumes.")
        if (state_dict["variant"] != self.cfg.gpo.variant
                or state_dict["normalize_latent"] != self.cfg.gpo.normalize_latent):
            raise ValueError("GPO checkpoint variant/latent normalization does not match configuration.")
        for name, module in self.named_children():
            module.load_state_dict(state_dict[name], strict=strict)
        for name in self.optimizer_names:
            getattr(self, name).load_state_dict(state_dict["optimizers"][name])
        self.num_updates = state_dict["num_updates"]
        self.lr_policy = state_dict["lr_policy"]
        self.penalty_coef = state_dict["penalty_coef"]
        self.env.set_progress(self.num_updates)
        return []
