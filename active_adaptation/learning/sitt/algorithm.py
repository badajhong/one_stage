"""VAIC adaptation of Student-Informed Teacher Training (ICLR 2025).

Teacher, student, and proxy retain their respective VAIC encoders and feed one
shared action decoder, following the paper's architecture and gradient rules.
"""

import copy
import weakref
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq, TensorDictModuleBase

from ..modules.rnn import recurrent_mode, set_recurrent_mode
from ..ppo.common import CatTensors, init_, make_batch, make_mlp, soft_copy_
from ..ppo.ppo_vel import (
    PPOVEL, OBJECT_KEY, OBJECT_GEO_KEY, OBJECT_TRANS_KEY, OBJECT_PRED_KEY,
    OBJECT_PRED_TRANS_KEY, PRIV_FEATURE_KEY, PRIV_PRED_KEY,
)
from .losses import action_kl, frozen, ramp_factor, shared_covariance_kl
from .networks import ProxyStudent


class JointRollout(TensorDictModuleBase):
    """Advance the student history, but only execute/sample the teacher policy."""

    def __init__(self, policy):
        super().__init__()
        self._owner = weakref.ref(policy)
        self.in_keys = list(policy.observation_spec.keys()) + ["is_init", "adapt_hx", "depth_hx"]
        self.out_keys = ["action", "sample_log_prob", "loc", "scale", "proxy_loc",
                         "proxy_scale", "sitt_kl", ("next", "adapt_hx"), ("next", "depth_hx")]

    def forward(self, td):
        policy = self._owner()
        teacher_td = policy.encode_teacher(td.copy())
        teacher_td[PRIV_PRED_KEY] = teacher_td[PRIV_FEATURE_KEY]
        policy.shared_actor(teacher_td)
        # get_dist() uses the same loc/scale names for all three paths. Separate
        # TensorDict containers prevent student/proxy writes corrupting PPO data.
        student_td = policy.encode_student(td.copy())
        proxy_dist = policy.proxy_distribution(td.copy())
        for key in ("action", "sample_log_prob", "loc", "scale"):
            td[key] = teacher_td[key]
        for key in ("adapt_hx", "depth_hx"):
            td["next", key] = student_td["next", key]
        td["proxy_loc"] = proxy_dist.mean.detach().clone()
        td["proxy_scale"] = proxy_dist.scale.detach().clone()
        td["sitt_kl"] = action_kl(
            td["loc"], td["scale"], td["proxy_loc"], td["proxy_scale"], policy.cfg.sitt.max_kl
        ).detach()
        return td


class TeacherRollout(TensorDictModuleBase):
    """Inference wrapper for the privileged encoder plus shared decoder."""

    def __init__(self, policy):
        super().__init__()
        self._owner = weakref.ref(policy)
        self.in_keys = list(policy.observation_spec.keys())
        self.out_keys = ["action", "sample_log_prob", "loc", "scale"]

    def forward(self, td):
        policy = self._owner()
        teacher_td = policy.encode_teacher(td.copy())
        teacher_td[PRIV_PRED_KEY] = teacher_td[PRIV_FEATURE_KEY]
        policy.shared_actor(teacher_td)
        for key in self.out_keys:
            td[key] = teacher_td[key]
        return td


class SITTPPOVEL(PPOVEL):
    checkpoint_version = 3
    objective_version = 2  # current-proxy KL, shared covariance gradient, paired alignment

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        self._validate_config(cfg, observation_spec)
        # Construct the original VAIC teacher and student perception layers.
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)

        # SITT architecture: teacher, student, and proxy use one decoder with
        # the student-safe [vel_command, policy, latent] input contract. The
        # two actors constructed by PPOVEL are replaced by actor_adapt only.
        self.shared_actor = self.actor_adapt
        del self.actor_adapt
        del self.actor
        del self.opt_adapt_actor
        policy_params = [
            {"params": self.shared_actor.parameters()},
            {"params": self.encoder_priv.parameters()},
        ]
        if hasattr(self, "height_cnn"):
            policy_params.append({"params": self.height_cnn.parameters()})
        self.opt_policy = torch.optim.Adam(policy_params, lr=self.lr_policy)
        # Teacher-training PPOVEL wires adaptation to GT objects. The deployed
        # student must instead use exactly the original finetune predicted path.
        old_cat = self.adapt_module[0]
        student_keys = [
            {OBJECT_KEY: OBJECT_PRED_KEY, OBJECT_TRANS_KEY: OBJECT_PRED_TRANS_KEY}.get(k, k)
            for k in old_cat.in_keys
        ]
        self.adapt_module = Seq(
            CatTensors(student_keys, "_adapt_inp", del_keys=False, sort=False),
            self.adapt_module[1],
            selected_out_keys=[PRIV_PRED_KEY, ("next", "adapt_hx")],
        )
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)

        # Same encoded teacher-observation contract and MLP shape as F_T, with
        # separately initialized parameters. Common policy/velocity inputs are
        # supplied to both paths by the shared decoder, not duplicated here.
        proxy_keys = list(self.encoder_priv[0].in_keys)
        self.proxy_student = ProxyStudent(proxy_keys, cfg.latent_dim).to(device)
        critic_keys = list(self.critic[0].in_keys)
        self.sitt_critic = Seq(
            CatTensors(critic_keys, "_sitt_critic_input", del_keys=False),
            Mod(nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1)),
                ["_sitt_critic_input"], ["sitt_value"]),
        ).to(device)
        fake = observation_spec.zero().to(device)
        self.encode_proxy(fake)
        # Match PPOVEL's privileged encoder initialization scheme while
        # drawing independent weights for the proxy.
        self.proxy_student.apply(init_)
        self.sitt_critic(fake)
        self.opt_proxy = torch.optim.Adam(self.proxy_student.parameters(), lr=cfg.sitt.proxy_lr)
        self.opt_sitt_critic = torch.optim.Adam(self.sitt_critic.parameters(), lr=cfg.lr)
        self.kl_reward_scale = float(env.step_dt) if cfg.sitt.kl_reward_mult_dt else 1.0

    @staticmethod
    def _validate_config(cfg, spec):
        if cfg.phase != "train" or not cfg.enable_residual_distillation:
            raise ValueError("SITT requires phase=train and enable_residual_distillation=true to construct VAIC's actor_adapt.")
        if cfg.adapt_module != "gru" or not cfg.use_depth or not cfg.use_object_adapt:
            raise ValueError("SITT currently supports the original depth/object/GRU VAIC architecture only.")
        if cfg.normalize_before_sum or cfg.train_dr_estimator:
            raise ValueError("SITT currently requires normalize_before_sum=false and train_dr_estimator=false.")
        if cfg.load_noise_scale is not None:
            raise ValueError("SITT requires load_noise_scale=null for faithful checkpoint restoration.")
        if cfg.train_every < 2 or cfg.num_minibatches < 1 or spec.shape[0] % cfg.num_minibatches:
            raise ValueError("Use train_every>=2 and num_envs divisible by num_minibatches for recurrent alignment.")
        for key in ("depth", "vel_command", "policy", "priv", "object_", "object_geo_", "ref_joint_pos_"):
            if key not in spec:
                raise ValueError(f"SITT missing observation {key!r}; use task=G1/sitt/skateboard.")
        settings = cfg.sitt
        if min(settings.reward_kl_coef, settings.policy_kl_coef, settings.warmup_iters,
               settings.ramp_iters, settings.proxy_action_coef, settings.student_action_coef) < 0:
            raise ValueError("SITT coefficients and schedule lengths must be nonnegative.")
        if settings.alignment_epochs < 1 or settings.proxy_epochs < 1 or not 0 < settings.ema_tau <= 1:
            raise ValueError("SITT alignment epochs must be positive and ema_tau in (0, 1].")
        if settings.max_kl is not None and settings.max_kl <= 0:
            raise ValueError("sitt.max_kl must be null or positive.")
        if settings.eval_policy not in ("student", "teacher"):
            raise ValueError("sitt.eval_policy must be student or teacher.")

    @property
    def sitt_factor(self):
        return ramp_factor(self.num_updates, self.cfg.sitt.warmup_iters, self.cfg.sitt.ramp_iters)

    def encode_teacher(self, td):
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            self.height_encoder(td)
        return self.encoder_priv(td)

    def encode_student(self, td, ema=False):
        # The original GRU step path expects a reset hidden state from the primer.
        # Mask explicitly too, so standalone replay/inference obeys is_init.
        if not recurrent_mode():
            for key in ("adapt_hx", "depth_hx"):
                td[key] = td[key] * (~td["is_init"]).to(td[key].dtype)
        suffix = "_ema" if ema else ""
        getattr(self, "temporal_depth_gru" + suffix)(td)
        getattr(self, "object_adapt" + suffix)(td)
        self.object_pred_transform(td)
        (self.adapt_ema if ema else self.adapt_module)(td)
        return td

    def proxy_distribution(self, td):
        self.encode_proxy(td)
        td[PRIV_PRED_KEY] = td["proxy_feature"]
        return self.shared_actor.get_dist(td)

    def encode_proxy(self, td):
        """Build the same derived privileged inputs used by encoder_priv."""
        self.object_transform(td)
        if hasattr(self, "height_encoder"):
            # Height preprocessing belongs to the existing teacher perception
            # path and is never trained by the proxy objective.
            with torch.no_grad():
                self.height_encoder(td)
        return self.proxy_student(td)

    @property
    def action_std(self):
        return self.shared_actor.module[0][2].module.actor_std

    def direct_policy_kl(self, td):
        """Current teacher vs current proxy, with SITT's gradient boundaries.

        The proxy encoder is fixed during PPO, but its decoded action changes
        whenever the shared decoder changes. Recompute it on every minibatch;
        rollout proxy_loc/proxy_scale are only historical diagnostics.
        Freeze the mean decoder, then use the live common std outside that
        context to implement Eq. (10)'s covariance gradient explicitly.
        """
        teacher = self.encode_teacher(td.copy())
        teacher[PRIV_PRED_KEY] = teacher[PRIV_FEATURE_KEY]
        with frozen(self.shared_actor):
            with torch.no_grad():
                proxy_mean = self.proxy_distribution(td.copy()).mean
            teacher_mean = self.shared_actor.get_dist(teacher).mean
        return shared_covariance_kl(
            teacher_mean, proxy_mean, self.action_std, self.cfg.sitt.max_kl,
        )

    def get_rollout_policy(self, mode="train"):
        if mode == "train":
            return JointRollout(self)
        if mode == "teacher" or (mode == "eval" and self.cfg.sitt.eval_policy == "teacher"):
            return TeacherRollout(self)
        if mode not in ("eval", "deploy", "student"):
            raise ValueError(f"Unknown SITT rollout mode: {mode}")
        return Seq(
            self.temporal_depth_gru_ema, self.object_adapt_ema, self.object_pred_transform,
            self.adapt_ema, self.shared_actor,
            selected_out_keys=["action", "sample_log_prob", "loc", "scale", PRIV_PRED_KEY,
                               ("next", "adapt_hx"), ("next", "depth_hx")],
        )

    @torch.no_grad()
    def _compute_advantage(self, td, critic, adv_key="adv", ret_key="ret", update_value_norm=True):
        # Keep the task returns, normalization and group weights exactly as VAIC.
        super()._compute_advantage(td, critic, adv_key, ret_key, update_value_norm)
        penalty_reward = -self.kl_reward_scale * td["sitt_kl"]
        penalty_adv, penalty_ret = self.gae(
            penalty_reward, td["next", "terminated"], td["next", "done"],
            td["sitt_value"], td["next", "sitt_value"], td["next", "discount"],
        )
        td["sitt_ret"] = penalty_ret
        coefficient = self.cfg.sitt.reward_kl_coef * self.sitt_factor
        if coefficient:
            # In VAIC's default branch adv_before_norm already includes group weights.
            raw_adv = td["adv_before_norm"].sum(dim=-1, keepdim=True)
            combined = raw_adv + coefficient * penalty_adv
            td[adv_key] = (combined - combined.mean(dim=(0, 1))) / (combined.std(dim=(0, 1)) + 1e-8)
        return td

    def _update_ppo(self, td):
        # A local adaptation of PPOVEL._update_ppo; the baseline is untouched.
        old_dist = self.dist_cls(loc=td["loc"].clone(), scale=td["scale"].clone())
        teacher_td = self.encode_teacher(td.copy())
        teacher_td[PRIV_PRED_KEY] = teacher_td[PRIV_FEATURE_KEY]
        dist = self.shared_actor.get_dist(teacher_td)
        valid = (td["step_count"] > 1).squeeze(-1)
        if not valid.any():
            raise ValueError("No valid teacher PPO samples; increase rollout length.")
        log_ratio = (dist.log_prob(td["action"]) - td["sample_log_prob"]).unsqueeze(-1)
        ratio = log_ratio.exp()
        surr1 = td["adv"] * ratio
        surr2 = td["adv"] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        if self.cfg.normalize_ratio:
            denominator = ratio.clamp(1 - self.clip_param, 1 + self.clip_param).detach()
            surr1, surr2 = surr1 / denominator, surr2 / denominator
        policy_loss = -torch.minimum(surr1, surr2)[valid].mean()
        entropy = dist.entropy().mean()
        values = self.critic(td.copy())["state_value"]
        value_loss = self.critic_loss_fn(td["ret"], values)[valid].mean(dim=0)
        sitt_values = self.sitt_critic(td.copy())["sitt_value"]
        sitt_value_loss = F.mse_loss(sitt_values[valid], td["sitt_ret"][valid])
        teacher_kl = self.direct_policy_kl(td)[valid].mean()
        loss = (policy_loss - self.entropy_coef * entropy + value_loss.mean() + sitt_value_loss
                + self.cfg.sitt.policy_kl_coef * self.sitt_factor * teacher_kl)
        self.opt_policy.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_sitt_critic.zero_grad(set_to_none=True)
        loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.cfg.max_grad_norm)
        encoder_norm = nn.utils.clip_grad_norm_(self.encoder_priv.parameters(), self.cfg.max_grad_norm)
        if hasattr(self, "height_cnn"):
            nn.utils.clip_grad_norm_(self.height_cnn.parameters(), self.cfg.max_grad_norm)
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.sitt_critic.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()
        self.opt_critic.step()
        self.opt_sitt_critic.step()
        info = {
            "actor/policy_loss": policy_loss.detach(), "actor/entropy": entropy.detach(),
            "actor/kl": kl_divergence(old_dist, dist).mean().detach(),
            "actor/grad_norm": actor_norm, "actor/priv_grad_norm": encoder_norm,
            "actor/clamp_ratio": ((ratio - 1).abs() > self.clip_param).float().mean().detach(),
            "critic/grad_norm": critic_norm, "sitt/teacher_proxy_kl": teacher_kl.detach(),
            "sitt/critic_loss": sitt_value_loss.detach(),
        }
        for i, group in enumerate(self.reward_groups):
            info[f"critic/{group}.value_loss"] = value_loss[i].detach()
        return info

    def _student_alignment_step(self, batch):
        # Called in recurrent mode with the entire decoder (including std) frozen.
        with torch.no_grad():
            teacher = self.encode_teacher(batch.copy())
            teacher_feature = teacher[PRIV_FEATURE_KEY]
            teacher[PRIV_PRED_KEY] = teacher_feature
            teacher_dist = self.shared_actor.get_dist(teacher)
            teacher_mean = teacher_dist.mean
        student = self.encode_student(batch.copy())
        mask = (~batch["is_init"]).to(student[PRIV_PRED_KEY].dtype)
        # Preserve VAIC's object auxiliary supervision. Paired feature/output
        # L1 follows the public SITT align_policy implementation.
        object_loss = ((student[OBJECT_PRED_KEY] - batch[OBJECT_KEY]).square() * mask).mean()
        priv_loss = ((student[PRIV_PRED_KEY] - teacher_feature).abs() * mask).mean()
        dist_student = self.shared_actor.get_dist(student)
        action_loss = ((dist_student.mean - teacher_mean).abs() * mask).mean()
        loss = object_loss + priv_loss + self.cfg.sitt.student_action_coef * action_loss
        self.opt_adapt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in self.opt_adapt.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        self.opt_adapt.step()
        return {"student/object_loss": object_loss.item(),
                "student/priv_loss": priv_loss.item(),
                "student/action_loss": action_loss.item(),
                "student/teacher_kl": action_kl(
                    teacher_mean, teacher_dist.scale, dist_student.mean, dist_student.scale,
                ).mean().item()}

    def _proxy_alignment_step(self, batch):
        # Recompute the student's target AFTER its paired update. GT object is
        # proxy input, not its target; the proxy imitates the current student.
        with torch.no_grad():
            student = self.encode_student(batch.copy())
            target_feature = student[PRIV_PRED_KEY]
            target_mean = self.shared_actor.get_dist(student).mean
        proxy_td = self.encode_proxy(batch.copy())
        proxy_feature = proxy_td["proxy_feature"]
        feature_loss = F.l1_loss(proxy_feature, target_feature)
        proxy_td[PRIV_PRED_KEY] = proxy_feature
        proxy_mean = self.shared_actor.get_dist(proxy_td).mean
        action_loss = F.l1_loss(proxy_mean, target_mean)
        loss = feature_loss + self.cfg.sitt.proxy_action_coef * action_loss
        self.opt_proxy.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.proxy_student.parameters(), self.cfg.max_grad_norm)
        self.opt_proxy.step()
        return {"proxy/feature_loss": feature_loss.item(), "proxy/action_loss": action_loss.item()}

    @set_recurrent_mode(True)
    def align_encoders(self, td, *, student=True, proxy=True):
        """Algorithm 1: update student, then proxy, on each paired minibatch.

        Defaults use equal epoch counts. Unequal counts remain available for
        existing experiment configs, with the remaining epochs updating one side.
        """
        student_infos, proxy_infos = [], []
        student_epochs = self.cfg.sitt.alignment_epochs if student else 0
        proxy_epochs = self.cfg.sitt.proxy_epochs if proxy else 0
        with frozen(self.shared_actor):
            for epoch in range(max(student_epochs, proxy_epochs)):
                for batch in make_batch(td, self.cfg.num_minibatches, self.cfg.train_every):
                    if epoch < student_epochs:
                        student_infos.append(self._student_alignment_step(batch))
                    if epoch < proxy_epochs:
                        proxy_infos.append(self._proxy_alignment_step(batch))
        info = self._mean_infos(student_infos) if student_infos else {}
        if proxy_infos:
            info.update(self._mean_infos(proxy_infos))
        return info

    def align_student(self, td):
        return self.align_encoders(td, proxy=False)

    def align_proxy(self, td):
        return self.align_encoders(td, student=False)

    @staticmethod
    def _mean_infos(infos):
        return {key: sum(info[key] for info in infos) / len(infos) for key in infos[0]}

    def train_op(self, td):
        td = td.exclude("stats")
        info = self.train_policy(td.copy())
        info.update(self.align_encoders(td))
        tau = self.cfg.sitt.ema_tau
        soft_copy_(self.adapt_module, self.adapt_ema, tau)
        soft_copy_(self.object_adapt, self.object_adapt_ema, tau)
        soft_copy_(self.temporal_depth_gru, self.temporal_depth_gru_ema, tau)
        info["sitt/reward_kl_coef"] = self.cfg.sitt.reward_kl_coef * self.sitt_factor
        info["sitt/policy_kl_coef"] = self.cfg.sitt.policy_kl_coef * self.sitt_factor
        info["sitt/rollout_kl"] = td["sitt_kl"].mean().item()
        info["sitt/penalty_reward"] = -self.kl_reward_scale * info["sitt/rollout_kl"]
        info["sitt/weighted_penalty_reward"] = info["sitt/reward_kl_coef"] * info["sitt/penalty_reward"]
        info["actor_std/mean"] = self.action_std.detach().mean().item()
        self.num_updates += 1
        return info

    def state_dict(self):
        # Unlike the legacy serializer, saving does not mutate student/EMA weights.
        state = OrderedDict((name, module.state_dict()) for name, module in self.named_children())
        state.update(sitt_version=self.checkpoint_version, sitt_objective_version=self.objective_version,
                     last_phase="sitt",
                     last_iter=self.num_updates, num_updates=self.num_updates, lr_policy=self.lr_policy)
        state["optimizers"] = {
            name: getattr(self, name).state_dict() for name in self.optimizer_names
        }
        return state

    optimizer_names = ("opt_policy", "opt_critic", "opt_adapt", "opt_proxy", "opt_sitt_critic")

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get("sitt_version") != self.checkpoint_version:
            raise ValueError("Expected an SITT checkpoint. Legacy teacher/finetune checkpoints are not SITT resumes.")
        for name, module in self.named_children():
            module.load_state_dict(state_dict[name], strict=strict)
        for name in self.optimizer_names:
            getattr(self, name).load_state_dict(state_dict["optimizers"][name])
        self.num_updates = state_dict["num_updates"]
        self.lr_policy = state_dict["lr_policy"]
        self.env.set_progress(self.num_updates)
        return []
