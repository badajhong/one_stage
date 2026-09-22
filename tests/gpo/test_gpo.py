"""GPO math, actual VAIC networks, recurrent collection and checkpoint tests."""

import copy
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs.utils import set_exploration_type, ExplorationType

from active_adaptation.learning.cts.algorithm import CTSPPOVEL
from active_adaptation.learning.cts.config import CTSPPOConfig
from active_adaptation.learning.gpo.algorithm import GPOPPOVEL
from active_adaptation.learning.gpo.config import GPOPPOConfig, GPOSettings
from active_adaptation.learning.gpo.losses import adapt_penalty, policy_losses
from active_adaptation.learning.gpo.runner import RolloutCollector, EpisodeMetrics, evaluate_learner, validate_resume
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.modules.rnn import set_recurrent_mode


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("GPO_TEST_DEVICE", "cpu")


class FakeEnv:
    def __init__(self, n=4):
        self.num_envs = n
        self.device = torch.device(DEVICE)
        self.current_iter = 0
        self.cfg = SimpleNamespace(reward={"tracking": {}, "feet": {}})
        self.action_manager = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(6)])
        shapes = {"command": (10,), "policy": (20,), "priv": (24,), "object_": (21,),
                  "object_geo_": (384,), "ref_joint_pos_": (6,), "vel_command": (25,),
                  "depth": (1, 36, 64)}
        self.observation_spec = Composite({k: Unbounded((n, *v), device=self.device)
                                           for k, v in shapes.items()}, shape=(n,), device=self.device)
        self.action_spec = Unbounded((n, 6), device=self.device)
        self.reward_spec = Unbounded((n, 2), device=self.device)

    def set_progress(self, n):
        self.current_iter = n


def make_policy(n=4, **settings):
    env = FakeEnv(n)
    cfg = GPOPPOConfig(train_every=8, ppo_epochs=1, num_minibatches=2)
    for k, v in settings.items():
        setattr(cfg.gpo, k, v)
    return GPOPPOVEL(cfg, env.observation_spec, env.action_spec, env.reward_spec, env.device, env)


def observation(p):
    td = p.observation_spec.rand()
    n = td.shape[0]
    td["is_init"] = torch.zeros(n, 1, dtype=torch.bool, device=p.device)
    td["step_count"] = torch.full((n, 1), 10, dtype=torch.long, device=p.device)
    td["adapt_hx"] = torch.randn(n, p.cfg.latent_dim, device=p.device)
    td["depth_hx"] = torch.randn(n, 64, device=p.device)
    return td


def rollout(p):
    env = p.env
    env.reset = lambda: observation(p)
    env.terminal_values = []

    @torch.no_grad()
    def step(carry):
        # Verify EVERY action was drawn from the guider distribution.
        g = p.shared_actor.get_dist(p.encode_guider(carry.copy()))
        torch.testing.assert_close(carry["loc"], g.loc)
        torch.testing.assert_close(carry["sample_log_prob"], g.log_prob(carry["action"]))
        nxt = observation(p)
        done = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=p.device)
        done[-1] = True  # timeout; last env resets every step
        nxt.update({"reward": torch.rand(env.num_envs, 2, device=p.device), "done": done,
                    "terminated": torch.zeros_like(done), "discount": torch.ones(env.num_envs, 1, device=p.device)})
        for key in ("adapt_hx", "depth_hx"):
            nxt[key] = carry["next", key]
        nxt["stats"] = TensorDict({"success": torch.ones(env.num_envs, 1, device=p.device),
                                   "episode_len": torch.full((env.num_envs, 1), 10., device=p.device)}, [env.num_envs])
        env.terminal_values.append(p.critic(p.encode_guider(nxt.clone()))["state_value"].clone())
        td = carry.copy()
        td["next"] = nxt
        reset = nxt.clone()
        for key in p.observation_spec.keys():
            reset[key][-1].zero_()
        for key in ("adapt_hx", "depth_hx"):
            reset[key][-1].zero_()
        reset["is_init"][-1] = True
        return td, reset

    env.step_and_maybe_reset = step
    return RolloutCollector(env, p).collect().clone()


class TestLosses(unittest.TestCase):
    def distributions(self):
        g_loc = torch.tensor([[.4, -.2], [-.3, .5], [.02, .01]], requires_grad=True)
        l_loc = torch.tensor([[-.4, .1], [.2, -.4], [.01, .02]], requires_grad=True)
        g_scale = torch.tensor([[.8, 1.2]] * 3, requires_grad=True)
        l_scale = torch.tensor([[1.1, .7]] * 3, requires_grad=True)
        return IndependentNormal(g_loc, g_scale), IndependentNormal(l_loc, l_scale)

    def test_reference_equations_both_variants_and_signed_advantages(self):
        g, l = self.distributions()
        a = torch.tensor([[.2, -.1], [-.2, .3], [.1, .1]])
        old_log = torch.tensor([-1.9, -1.4, -2.1])
        adv = torch.tensor([1.5, -2., .4])
        valid = torch.ones(3, dtype=torch.bool)
        for variant in ("clip", "penalty"):
            cfg = GPOSettings(variant=variant)
            actual = policy_losses(g, l, a, old_log, adv, valid, .2, cfg, .7)
            gl, ll = g.log_prob(a), l.log_prob(a)
            gr, lr = (gl - old_log).exp(), (ll - old_log).exp()
            kl = (torch.log(l.scale / g.scale) + (g.scale.square() + (g.loc - l.loc).square())
                  / (2 * l.scale.square()) - .5).mean(-1)
            if variant == "clip":
                clipped = ((gl - ll).exp().clamp(.9, 1.1) * lr).clamp(.8, 1.2)
                mask = ((gl - ll).exp() < .9) | ((gl - ll).exp() > 1.1)
                bk, alpha = (mask * kl).mean(), 2.
            else:
                clipped, bk, alpha = gr.clamp(.8, 1.2), .7 * kl.mean(), .7
            gp = -torch.minimum(gr * adv, clipped * adv).mean()
            lp = -torch.minimum(lr * adv, lr.clamp(.8, 1.2) * adv).mean()
            torch.testing.assert_close(actual["policy"], gp + bk + kl.mean() + alpha * lp)
            torch.testing.assert_close(actual["backtrack"], bk)
            torch.testing.assert_close(actual["alignment_kl"], kl.mean())

    def test_detach_directions_include_scales(self):
        for term, live in (("imitation", "learner"), ("backtrack", "guider"), ("guider_ppo", "guider"), ("learner_ppo", "learner")):
            g, l = self.distributions()
            losses = policy_losses(g, l, torch.ones(3, 2), torch.zeros(3), torch.ones(3),
                                    torch.ones(3, dtype=torch.bool), .2, GPOSettings(variant="penalty"), .7)
            leaves = (g.loc, g.scale, l.loc, l.scale)
            grads = torch.autograd.grad(losses[term], leaves, allow_unused=True)
            frozen = grads[2:] if live == "guider" else grads[:2]
            active = grads[:2] if live == "guider" else grads[2:]
            self.assertTrue(all(x is None for x in frozen), term)
            self.assertTrue(all(x is not None and x.abs().sum() > 0 for x in active), term)

    def test_double_clip_stops_guider_inside_outer_clip_and_mask(self):
        # One Gaussian sample, mu/pi=2 but mu/beta=1: only INNER clip binds.
        for sign in (1., -1.):
            g_loc = torch.tensor([[.2]], requires_grad=True)
            l_loc = torch.tensor([[1.2 if sign > 0 else .01]], requires_grad=True)
            g = IndependentNormal(g_loc, torch.tensor([[.5]]))
            l = IndependentNormal(l_loc, torch.tensor([[.5]]))
            a = torch.zeros(1, 1)
            cfg = GPOSettings(inner_clip=.01)
            terms = policy_losses(g, l, a, g.log_prob(a).detach(), torch.tensor([sign]),
                                  torch.ones(1, dtype=torch.bool), .2, cfg)
            grad = torch.autograd.grad(terms["guider_ppo"], g_loc, retain_graph=True)[0]
            torch.testing.assert_close(grad, torch.zeros_like(grad))
            self.assertEqual(terms["backtrack_fraction"].item(), 1.)
        g, _ = self.distributions()
        terms = policy_losses(g, g, torch.zeros(3, 2), torch.zeros(3), torch.ones(3),
                              torch.ones(3, dtype=torch.bool), .2, GPOSettings())
        self.assertEqual(terms["backtrack"].item(), 0.)
        self.assertEqual(terms["backtrack_fraction"].item(), 0.)

    def test_all_invalid_and_extreme_learner_ratio_are_finite(self):
        g, l = self.distributions()
        terms = policy_losses(g, l, torch.ones(3, 2), torch.full((3,), -1000.), torch.ones(3),
                              torch.zeros(3, dtype=torch.bool), .2, GPOSettings())
        self.assertTrue(all(torch.isfinite(x) for x in terms.values()))
        self.assertEqual(terms["policy"].item(), 0.)
        terms["policy"].backward()
        self.assertTrue(torch.isfinite(g.loc.grad).all())

    def test_penalty_schedule(self):
        cfg = GPOSettings()
        self.assertEqual(adapt_penalty(0., 0., cfg), 0.)
        self.assertAlmostEqual(adapt_penalty(0., .1, cfg), .12)
        self.assertAlmostEqual(adapt_penalty(1.2, 0., cfg), 1.)
        self.assertEqual(adapt_penalty(10., .1, cfg), 10.)


class TestNetworks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.p = make_policy()

    def test_exact_cts_network_parity_and_no_height(self):
        p = self.p
        cfg = CTSPPOConfig(train_every=8, ppo_epochs=1, num_minibatches=2)
        cts = CTSPPOVEL(cfg, p.observation_spec, p.env.action_spec, p.env.reward_spec, p.device, p.env)
        for name in ("encoder_priv", "object_transform", "temporal_depth_gru", "object_adapt",
                     "object_pred_transform", "adapt_module", "shared_actor"):
            getattr(cts, name).load_state_dict(getattr(p, name).state_dict())
        self.assertFalse(hasattr(p, "height_cnn"))
        self.assertEqual(list(p.encoder_priv[0].in_keys), ["priv", "object_", "object_trans"])
        obs = observation(p)
        with torch.no_grad():
            torch.testing.assert_close(p.encode_guider(obs.clone())["priv_feature"], cts.encode_teacher(obs.clone())["priv_feature"])
            torch.testing.assert_close(p.encode_learner(obs.clone())["priv_pred"], cts.encode_student(obs.clone())["priv_pred"])
        ids = [id(x) for g in p.opt_policy.param_groups for x in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_rollout_guider_all_envs_and_learner_hidden_state(self):
        p = self.p
        obs = observation(p)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            actual = p.get_rollout_policy("train")(obs.clone())
            expected = p.shared_actor(p.encode_guider(obs.clone()))
            learner = p.encode_learner(obs.clone())
            torch.testing.assert_close(actual["action"], expected["action"])
            for key in ("adapt_hx", "depth_hx"):
                torch.testing.assert_close(actual["next", key], learner["next", key])
        data = rollout(p)
        torch.testing.assert_close(data["next", "state_value"], torch.stack(p.env.terminal_values, 1))

    def test_recurrent_replay_matches_online_including_resets(self):
        p = self.p
        # Different CNN/MLP batch shapes can choose different TF32 kernels.
        # Test sequence/reset semantics at full precision, not that rounding.
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
                data = rollout(p)
                online = torch.stack([p.encode_learner(data[:, i].clone())["priv_pred"] for i in range(8)], 1)
                with set_recurrent_mode(True):
                    recurrent = p.encode_learner(data.clone())["priv_pred"]
        finally:
            torch.set_float32_matmul_precision(precision)
        torch.testing.assert_close(online, recurrent, atol=2e-6, rtol=1e-5)

    def test_learner_has_no_privileged_inputs_and_exports(self):
        p = self.p
        obs = observation(p)
        corrupt = obs.clone()
        for key in ("priv", "object_", "command", "ref_joint_pos_"):
            corrupt[key].fill_(float("nan"))
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            expected = p.get_rollout_policy("learner")(obs.clone())["action"]
            for mode in ("eval", "deploy"):
                actual = p.get_rollout_policy(mode)(corrupt.clone())
                torch.testing.assert_close(expected, actual["action"])
            deploy = copy.deepcopy(p.get_rollout_policy("deploy")).cpu()
            self.assertFalse(hasattr(deploy, "_owner"))
            self.assertFalse(any("critic" in k or "encoder_priv" in k for k in deploy.state_dict()))
            stream = io.BytesIO()
            torch.save(deploy, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
            safe = obs.cpu().select(*deploy.in_keys)
            torch.testing.assert_close(deploy(safe.clone())["action"], restored(safe.clone())["action"])

    def test_imitation_trains_entire_perception_and_not_guider_encoder(self):
        p = self.p
        data = rollout(p)
        p.zero_grad(set_to_none=True)
        with set_recurrent_mode(True):
            g = p.shared_actor.get_dist(p.encode_guider(data.copy()))
            l = p.shared_actor.get_dist(p.encode_learner(data.copy()))
            loss = policy_losses(g, l, data["action"], data["sample_log_prob"], torch.ones(4, 8, device=p.device),
                                  torch.ones(4, 8, dtype=torch.bool, device=p.device), .2, p.cfg.gpo)["imitation"]
            loss.backward()
        for module in (p.depth_cnn, p.temporal_depth_gru.gru, p.object_adapt, p.adapt_module, p.shared_actor):
            self.assertGreater(sum(x.grad.abs().sum().item() for x in module.parameters() if x.grad is not None), 0.)
        for module in (p.encoder_priv, p.critic):
            self.assertTrue(all(x.grad is None for x in module.parameters()))
        p.zero_grad(set_to_none=True)
        p.critic(p.encode_guider(observation(p)))["state_value"].sum().backward()
        for module in (p.encoder_priv, p.temporal_depth_gru, p.object_adapt, p.adapt_module, p.shared_actor):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_joint_update_and_exact_checkpoint_continuation(self):
        for variant in ("clip", "penalty"):
            p = make_policy(variant=variant, penalty_init=.1)
            before = {name: next(module.parameters()).detach().clone() for name, module in
                      (("encoder_priv", p.encoder_priv), ("depth_cnn", p.depth_cnn), ("shared_actor", p.shared_actor))}
            info = p.train_op(rollout(p))
            self.assertTrue(all(math.isfinite(v) for v in info.values()))
            self.assertNotIn("adapt/priv_loss", info)
            # Match PPOVEL's existing optimization dashboard; internal GPO
            # alignment diagnostics must not leak into run.log().
            expected = {"actor/policy_loss", "actor/entropy", "actor/mean_std",
                        "actor/grad_norm", "actor/clamp_ratio", "actor/kl",
                        "actor/priv_grad_norm", "actor/approx_kl", "critic/grad_norm",
                        "actor/lr", "actor/entropy_coef", "actor_std/mean"}
            expected.update(f"actor_std/{name}" for name in p.joint_names)
            expected.update(f"critic/{name}.{metric}" for name in p.reward_groups for metric in
                            ("explained_var", "value_loss", "ret_mean", "ret_std", "neg_rew_ratio"))
            self.assertEqual(set(info), expected)
            for name, value in before.items():
                self.assertFalse(torch.equal(value, next(getattr(p, name).parameters())))
            restored = make_policy(variant=variant, penalty_init=.1)
            restored.load_state_dict(copy.deepcopy(p.state_dict()))
            self.assertEqual(restored.num_updates, 1)
            self.assertEqual(restored.penalty_coef, p.penalty_coef)
            data = rollout(p)
            torch.manual_seed(123)
            p.train_op(data.clone())
            torch.manual_seed(123)
            restored.train_op(data.clone())
            for a, b in zip(p.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=1e-7 if DEVICE == "cuda" else 0, rtol=1e-5 if DEVICE == "cuda" else 0)
            with self.assertRaisesRegex(ValueError, "GPO checkpoint"):
                restored.load_state_dict({"cts_version": 1})

    def test_single_env_eval_and_training_validation(self):
        p = make_policy(n=1)
        with torch.no_grad():
            self.assertEqual(p.get_rollout_policy("eval")(observation(p))["action"].shape, (1, 6))
        with self.assertRaisesRegex(ValueError, "divisible"):
            p.get_rollout_policy("train")


class TestConfigAndMetrics(unittest.TestCase):
    def test_hydra_command_task_parity_and_resume_validation(self):
        with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
            cfg = compose(config_name="train", overrides=["algo=gpo_ppo_vel_train", "task=G1/gpo/skateboard_general_tracking"])
            cts = compose(config_name="train", overrides=["algo=cts_ppo_vel_train", "task=G1/cts/skateboard_general_tracking"])
        for key in ("observation", "reward", "command", "action", "termination", "randomization", "camera_dr"):
            self.assertEqual(cfg.task[key], cts.task[key])
        self.assertNotIn("height", cfg.task.observation)
        self.assertEqual(cfg.task.num_envs, 1024)
        self.assertEqual(cfg.algo.gpo.variant, "clip")
        self.assertEqual(cfg.algo.runner, "gpo")
        saved = copy.deepcopy(cfg)
        cfg.algo.gpo.evaluate_at_end = False
        validate_resume(saved, cfg)
        cfg.algo.gpo.inner_clip = .3
        with self.assertRaisesRegex(ValueError, "saved task"):
            validate_resume(saved, cfg)

    def test_guider_training_metrics_and_learner_eval(self):
        data = TensorDict({"next": TensorDict({"done": torch.tensor([[True], [False]]),
                          "stats": TensorDict({"success": torch.tensor([[.5], [99.]])}, [2])}, [2])}, [2])
        metrics = EpisodeMetrics()
        metrics.add(data)
        self.assertEqual(metrics.pop(), {"train/stats/success": .5})
        self.assertEqual(metrics.pop(), {})
        roles = []
        p = SimpleNamespace(get_rollout_policy=lambda role: roles.append(role) or (lambda x: x))
        data["next", "stats", "episode_len"] = torch.ones(2, 1)
        env = SimpleNamespace(num_envs=2, device="cpu", max_episode_length=2, eval=lambda: None,
                              base_env=SimpleNamespace(eval=lambda: None), set_seed=lambda _: None,
                              reset=lambda: TensorDict({}, [2]), step_and_maybe_reset=lambda x: (data, x))
        info = evaluate_learner(env, p)
        self.assertEqual(roles, ["learner"])
        self.assertEqual(info["eval/success"], .5)
        self.assertEqual(info["episode_cnt"], 1)
        self.assertNotIn("eval/incomplete_episodes", info)


if __name__ == "__main__":
    unittest.main()
