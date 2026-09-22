"""CTS Algorithm 1 with the existing VAIC depth/object/GRU encoders.

Rollout -> teacher + student PPO -> student-only latent reconstruction.
Student latents are fixed throughout PPO; targets are paired privileged states
from the STUDENT trajectories. No SITT proxy or KL reward is constructed.
"""

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
from .networks import CTSRollout, LatentCritic, StudentRollout


class CTSPPOVEL(PPOVEL):
    checkpoint_version = 1
    optimizer_names = ("opt_policy", "opt_critic", "opt_adapt")

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_config(cfg, observation_spec)
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self.num_teacher = int(observation_spec.shape[0] * cfg.cts.teacher_ratio)
        self.num_student = observation_spec.shape[0] - self.num_teacher
        self.shared_actor = self.actor_adapt
        del self.actor_adapt, self.actor, self.opt_adapt_actor
        # CTS uses the online student for rollout, reconstruction and evaluation.
        # Switching to an EMA encoder at deployment changes the actor's input.
        del self.adapt_ema, self.object_adapt_ema, self.temporal_depth_gru_ema
        student_keys = [
            {"object_": "object_pred", "object_trans": "object_pred_trans"}.get(k, k)
            for k in self.adapt_module[0].in_keys
        ]
        self.adapt_module = Seq(
            CatTensors(student_keys, "_adapt_inp", del_keys=False, sort=False),
            self.adapt_module[1],
            selected_out_keys=["priv_pred", ("next", "adapt_hx")],
        )
        policy_params = list(self.shared_actor.parameters()) + list(self.encoder_priv.parameters())
        if hasattr(self, "height_cnn"):
            policy_params += list(self.height_cnn.parameters())
        self.opt_policy = torch.optim.Adam(policy_params, lr=cfg.lr)
        for group in self.opt_adapt.param_groups:
            group["lr"] = cfg.cts.student_lr

        critic_keys = list(self.critic[0].in_keys) + ["cts_latent"]
        self.critic = LatentCritic(Seq(
            CatTensors(critic_keys, "_cts_critic_input", del_keys=False),
            Mod(nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(len(self.reward_groups))),
                ["_cts_critic_input"], ["state_value"]),
        )).to(device)
        fake = observation_spec.zero().to(device)
        fake["cts_latent"] = torch.zeros(*fake.batch_size, cfg.latent_dim, device=device)
        self.critic(fake)
        # Match PPOVEL initialization for the new input layer and critic head.
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, .01)
                nn.init.zeros_(module.bias)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    @staticmethod
    def _validate_config(cfg, spec):
        if cfg.phase != "train" or not cfg.enable_residual_distillation:
            raise ValueError("CTS requires phase=train and enable_residual_distillation=true for VAIC construction.")
        if cfg.adapt_module != "gru" or not cfg.use_depth or not cfg.use_object_adapt:
            raise ValueError("CTS supports the VAIC depth/object/GRU perception architecture.")
        if cfg.normalize_before_sum or cfg.train_dr_estimator:
            raise ValueError("CTS requires normalize_before_sum=false and train_dr_estimator=false.")
        if cfg.load_noise_scale is not None:
            raise ValueError("CTS requires load_noise_scale=null for faithful checkpoint loading.")
        c = cfg.cts
        if not 0 < c.teacher_ratio < 1:
            raise ValueError("cts.teacher_ratio must be strictly between 0 and 1.")
        if cfg.train_every < 2 or cfg.num_minibatches < 1 or cfg.ppo_epochs < 1:
            raise ValueError("CTS needs train_every>=2 and positive PPO epochs/minibatches.")
        # Allow small evaluation env counts; training runner enforces valid groups.
        if c.reconstruction_epochs < 1 or not math.isfinite(c.student_lr) or c.student_lr <= 0:
            raise ValueError("CTS reconstruction epochs and student_lr must be positive.")
        if not math.isfinite(c.object_loss_coef) or c.object_loss_coef < 0:
            raise ValueError("CTS object_loss_coef must be finite and nonnegative.")
        if c.eval_policy not in ("teacher", "student"):
            raise ValueError("cts.eval_policy must be teacher or student.")
        for key in ("policy", "priv", "depth", "vel_command", "object_", "object_geo_", "ref_joint_pos_"):
            if key not in spec:
                raise ValueError(f"CTS missing observation {key!r}.")

    def validate_training_groups(self):
        for n in (self.num_teacher, self.num_student):
            if n < self.cfg.num_minibatches or n % self.cfg.num_minibatches:
                raise ValueError("Each CTS group must be nonempty and divisible by num_minibatches.")

    def normalize_latent(self, z):
        return F.normalize(z, p=2, dim=-1, eps=1e-6) if self.cfg.cts.normalize_latent else z

    def encode_teacher(self, td):
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        self.encoder_priv(td)
        td["priv_feature"] = self.normalize_latent(td["priv_feature"])
        return td

    def encode_student(self, td):
        if not recurrent_mode():
            for key in ("adapt_hx", "depth_hx"):
                td[key] = td[key] * (~td["is_init"]).to(td[key].dtype)
        self.temporal_depth_gru(td)
        self.object_adapt(td)
        self.object_pred_transform(td)
        self.adapt_module(td)
        td["priv_pred"] = self.normalize_latent(td["priv_pred"])
        return td

    def encode_role(self, td, role):
        if role == "train":
            nt = self.num_teacher
            teacher = self.encode_teacher(td[:nt].copy())
            student = self.encode_student(td[nt:].copy())
            td["priv_pred"] = torch.cat([teacher["priv_feature"], student["priv_pred"]], 0)
            for key in ("adapt_hx", "depth_hx"):
                td["next", key] = torch.cat([torch.zeros_like(td[key][:nt]), student["next", key]], 0)
        elif role == "teacher":
            self.encode_teacher(td)
            td["priv_pred"] = td["priv_feature"]
            for key in ("adapt_hx", "depth_hx"):
                td["next", key] = torch.zeros_like(td[key])
        elif role == "student":
            self.encode_student(td)
        else:
            raise ValueError(f"Unknown CTS role: {role}")
        td["cts_latent"] = td["priv_pred"]
        return td

    def get_rollout_policy(self, mode="train"):
        role = self.cfg.cts.eval_policy if mode == "eval" else mode
        if role == "deploy":
            role = "student"
        if role not in ("train", "teacher", "student"):
            raise ValueError(f"Unknown CTS mode: {mode}")
        if role == "train":
            self.validate_training_groups()
        if role == "student":
            return StudentRollout(self)
        return CTSRollout(self, role)

    @torch.no_grad()
    def _compute_advantage(self, td):
        super()._compute_advantage(td, self.critic)
        # Keep VAIC's reward-group aggregation, then normalize within each group.
        raw = td["adv_before_norm"].sum(-1, keepdim=True)
        for section in (slice(0, self.num_teacher), slice(self.num_teacher, None)):
            a = raw[section]
            td["adv"][section] = (a - a.mean()) / (a.std() + 1e-8)

    def _actor_loss(self, batch, role):
        local = batch.copy()
        if role == "teacher":
            self.encode_teacher(local)
            local["priv_pred"] = local["priv_feature"]
        else:
            # Student encoder stays fixed for all PPO epochs. Stored online
            # latents also avoid shuffled GRU replay and normalizer drift.
            local["priv_pred"] = batch["cts_latent"].detach()
        dist = self.shared_actor.get_dist(local)
        ratio = (dist.log_prob(batch["action"]) - batch["sample_log_prob"]).exp().unsqueeze(-1)
        s1 = batch["adv"] * ratio
        s2 = batch["adv"] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        if self.cfg.normalize_ratio:
            denominator = ratio.clamp(1 - self.clip_param, 1 + self.clip_param).detach()
            s1, s2 = s1 / denominator, s2 / denominator
        valid = (batch["step_count"] > 1).squeeze(-1)
        # A minibatch can contain only newly reset envs; retain a zero-gradient
        # loss instead of introducing NaNs or silently changing group weights.
        count = valid.sum().clamp_min(1)
        loss = -(torch.minimum(s1, s2).squeeze(-1) * valid).sum() / count
        entropy = dist.entropy().mean()
        old_dist = self.dist_cls(loc=batch["loc"], scale=batch["scale"])
        kl = kl_divergence(old_dist, dist).mean().detach()
        local["cts_latent"] = local["priv_pred"].detach()
        value_errors = (self.critic(local)["state_value"] - batch["ret"]).square()
        value_sum = (value_errors * valid.unsqueeze(-1)).sum(0)
        return loss, entropy, kl, value_sum, valid.sum(), ((ratio - 1).abs() > self.clip_param).float().mean().detach()

    def _update_ppo(self, teacher, student):
        tl, th, tk, tv, tn, _ = self._actor_loss(teacher, "teacher")
        sl, sh, sk, sv, sn, clip = self._actor_loss(student, "student")
        value_loss = ((tv + sv) / (tn + sn).clamp_min(1)).mean()
        entropy = (th * teacher.numel() + sh * student.numel()) / (teacher.numel() + student.numel())
        loss = tl + sl - self.entropy_coef * entropy + value_loss
        self.opt_policy.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.shared_actor.parameters()), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(list(self.encoder_priv.parameters()), self.cfg.max_grad_norm)
        if hasattr(self, "height_cnn"):
            nn.utils.clip_grad_norm_(self.height_cnn.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()
        self.opt_critic.step()
        # Control the shared learning rate using both branches, while publishing
        # only student distribution diagnostics to W&B.
        kl = torch.maximum(tk, sk).item()
        if self.desired_kl is not None:
            if kl > self.desired_kl * 2:
                self.lr_policy = max(1e-5, self.lr_policy / 1.5)
            elif 0 < kl < self.desired_kl / 2:
                self.lr_policy = min(1e-2, self.lr_policy * 1.5)
            for group in self.opt_policy.param_groups:
                group["lr"] = self.lr_policy
        info = {"actor/policy_loss": sl.detach().item(), "actor/entropy": sh.detach().item(),
                "actor/kl": sk.item(), "actor/clamp_ratio": clip.item()}
        for i, name in enumerate(self.reward_groups):
            info[f"critic/{name}.value_loss"] = (sv[i] / sn.clamp_min(1)).detach().item()
        return info

    def _student_step(self, batch):
        # Same student env and timestep on both sides, AFTER teacher PPO.
        with torch.no_grad():
            target = self.encode_teacher(batch.copy())["priv_feature"]
        student = self.encode_student(batch.copy())
        mask = (~batch["is_init"]).squeeze(-1)
        count = mask.sum().clamp_min(1)
        rec = ((student["priv_pred"] - target).square().sum(-1) * mask).sum() / count
        obj = ((student["object_pred"] - batch["object_"]).square().mean(-1) * mask).sum() / count
        loss = rec + self.cfg.cts.object_loss_coef * obj
        self.opt_adapt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in self.opt_adapt.param_groups for p in group["params"]]
        grad = nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        self.opt_adapt.step()
        # Log elementwise MSE like VAIC, without changing CTS's squared-L2
        # reconstruction objective (rec) used for backpropagation above.
        return {"adapt/priv_loss": rec.item() / target.shape[-1], "adapt/object_loss": obj.item(),
                "adapt/grad_norm": grad.item()}

    @set_recurrent_mode(True)
    def reconstruct_student(self, student):
        infos = []
        for _ in range(self.cfg.cts.reconstruction_epochs):
            for batch in make_batch(student, self.cfg.num_minibatches, self.cfg.train_every):
                infos.append(self._student_step(batch))
        return self._mean_infos(infos)

    @staticmethod
    def _mean_infos(infos):
        return {k: sum(info[k] for info in infos) / len(infos) for k in infos[0]}

    def train_op(self, td):
        self.validate_training_groups()
        td = td.exclude("stats")
        self._compute_advantage(td)
        progress = min(1., self.num_updates / max(1, self.cfg.entropy_decay_iters))
        self.entropy_coef = self.cfg.entropy_coef_start + progress * (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
        teacher, student = td[:self.num_teacher], td[self.num_teacher:]
        infos = []
        # Exclude depth from PPO minibatches: the fixed student latent was stored
        # during rollout, and the teacher has no depth input.
        for _ in range(self.cfg.ppo_epochs):
            tb = make_batch(teacher.exclude("depth"), self.cfg.num_minibatches)
            sb = make_batch(student.exclude("depth"), self.cfg.num_minibatches)
            for t, s in zip(tb, sb):
                infos.append(self._update_ppo(t, s))
        info = self._mean_infos(infos)
        info.update(self.reconstruct_student(student))
        info["actor/lr"] = self.lr_policy
        info["actor/entropy_coef"] = self.entropy_coef
        action_std = self.shared_actor.module[0][2].module.actor_std.detach()
        info["actor_std/mean"] = action_std.mean().item()
        for name, value in zip(self.env.action_manager.joint_names, action_std.tolist()):
            info[f"actor_std/{name}"] = value
        self.num_updates += 1
        return info

    def state_dict(self):
        state = OrderedDict((name, module.state_dict()) for name, module in self.named_children())
        state.update(cts_version=self.checkpoint_version, last_phase="cts", last_iter=self.num_updates,
                     num_updates=self.num_updates, lr_policy=self.lr_policy,
                     normalize_latent=self.cfg.cts.normalize_latent)
        state["optimizers"] = {name: getattr(self, name).state_dict() for name in self.optimizer_names}
        return state

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get("cts_version") != self.checkpoint_version:
            raise ValueError("Expected a CTS checkpoint; SITT/VAIC checkpoints are not CTS resumes.")
        if state_dict["normalize_latent"] != self.cfg.cts.normalize_latent:
            raise ValueError("CTS checkpoint latent normalization does not match configuration.")
        for name, module in self.named_children():
            module.load_state_dict(state_dict[name], strict=strict)
        for name in self.optimizer_names:
            getattr(self, name).load_state_dict(state_dict["optimizers"][name])
        self.num_updates = state_dict["num_updates"]
        self.lr_policy = state_dict["lr_policy"]
        self.env.set_progress(self.num_updates)
        return []
