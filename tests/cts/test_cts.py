"""Real VAIC module tests; no simulator or W&B account required."""

import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs.utils import set_exploration_type, ExplorationType

from active_adaptation.learning.cts.algorithm import CTSPPOVEL
from active_adaptation.learning.cts.config import CTSPPOConfig
from active_adaptation.learning.cts.runner import RolloutCollector, evaluate_student
from active_adaptation.learning.cts.metrics import EpisodeMetrics, RewardMetrics
from active_adaptation.learning.modules.rnn import set_recurrent_mode


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("CTS_TEST_DEVICE", "cpu")


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
    cfg = CTSPPOConfig(train_every=8, ppo_epochs=1, num_minibatches=2, desired_kl=None)
    cfg.cts.reconstruction_epochs = 1
    for k, v in settings.items():
        setattr(cfg.cts, k, v)
    return CTSPPOVEL(cfg, env.observation_spec, env.action_spec, env.reward_spec, env.device, env)


def observation(p):
    td = p.observation_spec.rand()
    n = td.shape[0]
    td["is_init"] = torch.zeros(n, 1, dtype=torch.bool, device=p.device)
    td["step_count"] = torch.full((n, 1), 10, dtype=torch.long, device=p.device)
    td["adapt_hx"] = torch.randn(n, p.cfg.latent_dim, device=p.device)
    td["depth_hx"] = torch.randn(n, 64, device=p.device)
    return td


def attach_steps(p):
    env = p.env
    env.reset = lambda: observation(p)
    recorded = []

    @torch.no_grad()
    def step(carry):
        nxt = observation(p)
        done = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=p.device)
        done[-1] = True  # student timeout; must bootstrap before resetting
        nxt.update({"reward": torch.rand(env.num_envs, 2, device=p.device), "done": done,
                    "terminated": torch.zeros_like(done), "discount": torch.ones(env.num_envs, 1, device=p.device)})
        for key in ("adapt_hx", "depth_hx"):
            nxt[key] = carry["next", key]
        nxt["stats"] = TensorDict({"success": torch.ones(env.num_envs, 1, device=p.device),
                                   "episode_len": torch.full((env.num_envs, 1), 10., device=p.device)}, [env.num_envs])
        recorded.append(p.critic(p.encode_role(nxt.clone(), "train"))["state_value"].clone())
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
    return recorded


def rollout(p):
    attach_steps(p)
    return RolloutCollector(p.env, p).collect().clone()


def weights(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def changed(before, module):
    return any(not torch.equal(before[k], v) for k, v in module.named_parameters())


class TestCTS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.p = make_policy()

    def test_split_actions_and_student_privilege_boundary(self):
        p = self.p
        obs = observation(p)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            actual = p.get_rollout_policy("train")(obs.clone())
            teacher = p.get_rollout_policy("teacher")(obs.clone())
            student = p.get_rollout_policy("student")(obs.clone())
            torch.testing.assert_close(actual["action"][:2], teacher["action"][:2])
            torch.testing.assert_close(actual["action"][2:], student["action"][2:])
            torch.testing.assert_close(actual["sample_log_prob"][2:], student["sample_log_prob"][2:])
            corrupt = obs.clone()
            for key in ("priv", "object_", "command", "ref_joint_pos_"):
                corrupt[key].fill_(float("nan"))
            # Deploy/eval must remain student-only even for the first half.
            for mode in ("eval", "deploy"):
                result = p.get_rollout_policy(mode)(corrupt.clone())
                torch.testing.assert_close(result["action"], student["action"])
            safe_obs = obs.select(*p.get_rollout_policy("student").in_keys)
            torch.testing.assert_close(p.get_rollout_policy("student")(safe_obs)["action"], student["action"])
            torch.testing.assert_close(actual["cts_latent"].norm(dim=-1), torch.ones(4, device=p.device))

    def test_student_ppo_gradients_only_reach_actor(self):
        p = self.p
        batch = rollout(p)
        p._compute_advantage(batch)
        student = batch[2:].reshape(-1)
        p.zero_grad(set_to_none=True)
        loss, entropy, kl, *_ = p._actor_loss(student, "student")
        self.assertLess(abs(kl.item()), 1e-5)
        (loss - .001 * entropy).backward()
        self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in p.shared_actor.parameters()))
        for module in (p.encoder_priv, p.temporal_depth_gru, p.object_adapt, p.adapt_module, p.critic):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_student_export_has_no_teacher_or_critic(self):
        p = self.p
        deploy = copy.deepcopy(p.get_rollout_policy("deploy")).cpu()
        self.assertFalse(any("encoder_priv" in k or "critic" in k for k in deploy.state_dict()))
        self.assertFalse(hasattr(deploy, "_owner"))
        obs = observation(p).cpu().select(*deploy.in_keys)
        stream = io.BytesIO()
        torch.save(deploy, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=False)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            torch.testing.assert_close(deploy(obs.clone())["action"], loaded(obs.clone())["action"])

    def test_teacher_ppo_and_critic_gradient_boundaries(self):
        p = self.p
        batch = rollout(p)
        p._compute_advantage(batch)
        teacher = batch[:2].reshape(-1)
        p.zero_grad(set_to_none=True)
        loss, *_ = p._actor_loss(teacher, "teacher")
        loss.backward()
        self.assertGreater(sum(x.grad.abs().sum().item() for x in p.encoder_priv.parameters() if x.grad is not None), 0)
        self.assertTrue(all(x.grad is None for x in p.adapt_module.parameters()))
        p.zero_grad(set_to_none=True)
        for role in ("teacher", "student"):
            td = p.encode_role(observation(p), role)
            p.critic(td)["state_value"].sum().backward()
        for module in (p.encoder_priv, p.temporal_depth_gru, p.object_adapt, p.adapt_module, p.shared_actor):
            self.assertTrue(all(x.grad is None for x in module.parameters()))
        self.assertTrue(any(x.grad is not None for x in p.critic.parameters()))

    def test_latent_reconstruction_trains_entire_student(self):
        p = self.p
        batch = rollout(p)[2:]
        p.cfg.cts.object_loss_coef = 0.  # isolate latent supervision
        before = weights(p)
        p.zero_grad(set_to_none=True)
        with set_recurrent_mode(True):
            p._student_step(batch)
        for module in (p.depth_cnn, p.temporal_depth_gru.gru, p.object_adapt, p.adapt_module):
            self.assertGreater(sum(x.grad.abs().sum().item() for x in module.parameters() if x.grad is not None), 0)
        for name, value in p.named_parameters():
            if name.startswith(("encoder_priv.", "shared_actor.", "critic.")):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(changed(before, p))

    def test_paired_target_uses_student_states_after_ppo(self):
        p = self.p
        data = rollout(p)
        teacher_before = weights(p.encoder_priv)
        original = p.encode_teacher
        seen = []

        def record(td):
            if td.ndim == 2:  # recurrent reconstruction only
                self.assertTrue(changed(teacher_before, p.encoder_priv))
                for row in td["priv"]:
                    self.assertTrue(any(torch.equal(row, source) for source in data["priv"][2:]))
                seen.append(True)
            return original(td)

        with patch.object(p, "encode_teacher", side_effect=record):
            info = p.train_op(data)
        self.assertEqual(len(seen), p.cfg.num_minibatches)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in info.values()))
        self.assertFalse(any("teacher" in key for key in info))
        self.assertFalse(any(key.startswith("student/") for key in info))
        for key in ("actor/policy_loss", "critic/tracking.value_loss", "adapt/priv_loss", "actor_std/mean"):
            self.assertIn(key, info)

    def test_groupwise_advantage_normalization(self):
        p = self.p
        data = rollout(p)
        data["next", "reward"][:2] += 100.
        p._compute_advantage(data)
        for a in (data["adv"][:2], data["adv"][2:]):
            self.assertLess(abs(a.mean().item()), 1e-5)
            self.assertAlmostEqual(a.std().item(), 1., places=5)

    def test_checkpoint_roundtrip_and_next_update(self):
        p = self.p
        p.train_op(rollout(p))
        state = copy.deepcopy(p.state_dict())
        restored = make_policy()
        restored.load_state_dict(state)
        self.assertEqual(restored.num_updates, 1)
        for a, b in zip(p.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        data = rollout(p)
        torch.manual_seed(123)
        p.train_op(data.clone())
        torch.manual_seed(123)
        restored.train_op(data.clone())
        for a, b in zip(p.parameters(), restored.parameters()):
            # CUDA convolution reductions can differ at roundoff level even
            # with the same RNG. The weights loaded above are still exact.
            torch.testing.assert_close(a, b, rtol=1e-5 if DEVICE == "cuda" else 0,
                                       atol=1e-7 if DEVICE == "cuda" else 0)
        with self.assertRaisesRegex(ValueError, "CTS checkpoint"):
            restored.load_state_dict({"sitt_version": 3})

    def test_terminal_bootstrap_is_not_reset_state(self):
        p = self.p
        recorded = attach_steps(p)
        collector = RolloutCollector(p.env, p)
        data = collector.collect().clone()
        torch.testing.assert_close(data["next", "state_value"], torch.stack(recorded, 1))
        self.assertTrue(data["is_init"][-1, 1:].all())
        collector.collect()
        torch.testing.assert_close(data["next", "state_value"], torch.stack(recorded[:8], 1))

    def test_eval_allows_one_env(self):
        p = make_policy(n=1)
        with torch.no_grad():
            self.assertEqual(p.get_rollout_policy("eval")(observation(p))["action"].shape, (1, 6))
        with self.assertRaisesRegex(ValueError, "group"):
            p.get_rollout_policy("train")


class TestMetricsAndConfig(unittest.TestCase):
    def test_student_only_statistics_with_teacher_sentinels(self):
        reward = torch.tensor([1000., 1000., 2., 4.]).view(4, 1, 1).expand(4, 3, 2).clone()
        success = torch.tensor([1., 1., 0., 0.]).view(4, 1, 1).expand(4, 3, 1).clone()
        done = torch.zeros(4, 3, 1, dtype=torch.bool)
        done[:, -1] = True
        data = TensorDict({"next": TensorDict({"reward": reward, "done": done,
                           "stats": TensorDict({"success": success, "episode_len": reward[..., :1]}, [4, 3])}, [4, 3])}, [4, 3])
        data["next", "stats", "tracking", "term"] = reward[..., :1]
        metrics = EpisodeMetrics(slice(2, None), "train/stats")
        teacher = EpisodeMetrics(slice(None, 2), "cts_teacher/episode")
        metrics.add(data)
        teacher.add(data)
        self.assertEqual(metrics.episodes, 2)
        info = metrics.pop()
        info.update(teacher.pop())
        self.assertEqual(info["train/stats/success"], 0.)
        self.assertEqual(info["train/stats/episode_len"], 3.)
        self.assertEqual(info["train/stats/tracking/term"], 3.)
        self.assertEqual(info["cts_teacher/episode/success"], 1.)
        self.assertEqual(info["cts_teacher/episode/tracking/term"], 1000.)
        self.assertEqual(metrics.pop(), {})
        data["next", "done"][2:] = False
        metrics.add(data)
        self.assertEqual(metrics.pop(), {})  # teacher-only endings never log episodes

    def test_reward_ema_matches_vaic_with_separate_active_counts(self):
        class Reward:
            weight = 2.
            calls = 0

            def compute(self):
                self.calls += 1
                return self.result

            def __call__(self):
                result = self.compute()
                if isinstance(result, tuple):
                    reward, active = result
                    return self.weight * (reward * active.float()), active.sum().item()
                return self.weight * result, result.numel()

        term = Reward()
        base = SimpleNamespace(num_envs=4, _stats_ema_decay=.99,
                               reward_groups={"tracking": SimpleNamespace(funcs={"term": term})})
        metrics = RewardMetrics(base, 2)
        try:
            # An active zero reward must still contribute to the denominator.
            rewards = torch.tensor([[1000.], [3000.], [0.], [4.]])
            active = torch.tensor([[True], [False], [True], [True]])
            term.result = (rewards, active)
            output, count = term()
            torch.testing.assert_close(output, 2 * rewards * active)
            self.assertEqual(count, 3)
            self.assertEqual(term.calls, 1)
            info = metrics.snapshot()
            self.assertEqual(info["reward.tracking/term"], 4.)
            self.assertEqual(info["cts_teacher/reward/tracking/term"], 2000.)
            term.result = (rewards, torch.tensor([[False], [True], [False], [True]]))
            term()
            info = metrics.snapshot()
            self.assertAlmostEqual(info["reward.tracking/term"], (8 * .99 + 8) / (2 * .99 + 1), places=5)
            self.assertAlmostEqual(info["cts_teacher/reward/tracking/term"],
                                   (2000 * .99 + 6000) / (.99 + 1), places=3)
            # Unmasked terms (including disabled debug terms) use numel counts.
            term.result = rewards
            term()
            self.assertEqual(term.calls, 3)
            self.assertFalse(any(k.startswith("student/") for k in info))
            torch.testing.assert_close(rewards, torch.tensor([[1000.], [3000.], [0.], [4.]]))
        finally:
            metrics.close()
        self.assertNotIn("compute", vars(term))
        metrics.close()  # exception/finally cleanup can run twice
        term()
        self.assertEqual(term.calls, 4)

    def test_inactive_group_does_not_log_fake_zero_reward(self):
        metrics = RewardMetrics(SimpleNamespace(num_envs=4, _stats_ema_decay=.99, reward_groups={}), 2)
        metrics.record(("feet", "term"), (torch.ones(4, 1), torch.tensor([[1], [1], [0], [0]])), 1.)
        self.assertEqual(metrics.snapshot(), {"cts_teacher/reward/feet/term": 1.})

    def test_evaluation_first_episode_and_incomplete_count(self):
        calls = []
        policy = SimpleNamespace(get_rollout_policy=lambda role: calls.append(role) or (lambda x: x))
        env = SimpleNamespace(num_envs=3, device="cpu", max_episode_length=2,
                              eval=lambda: None, set_seed=lambda _: None,
                              reset=lambda: TensorDict({}, [3]), base_env=SimpleNamespace(eval=lambda: None))
        def step(td):
            nxt = TensorDict({"done": torch.tensor([[True], [True], [False]]),
                              "stats": TensorDict({"success": torch.tensor([[1.], [0.], [99.]]),
                                                    "episode_len": torch.tensor([[2.], [4.], [1.]]),
                                                    "tracking": TensorDict({"term": torch.tensor([[8.], [8.], [99.]])}, [3])}, [3])}, [3])
            return TensorDict({"next": nxt}, [3]), td
        env.step_and_maybe_reset = step
        info = evaluate_student(env, policy)
        self.assertEqual(calls, ["student"])
        self.assertEqual(info["eval/success"], .5)
        self.assertEqual(info["eval/episode_len"], 3.)
        self.assertEqual(info["eval/tracking/term"], 3.)  # mean(8/2, 8/4), same as VAIC
        self.assertAlmostEqual(info["eval/tracking/term_std"], 2 ** .5, places=6)
        self.assertEqual(info["episode_cnt"], 2)
        self.assertEqual(info["eval/incomplete_episodes"], 1)
        self.assertFalse(any(k.startswith("student/") for k in info))

    def test_config_and_vaic_parity(self):
        with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
            cfg = compose(config_name="train", overrides=["algo=cts_ppo_vel_train", "task=G1/cts/skateboard_general_tracking"])
        original = OmegaConf.load(ROOT / "cfg/task/G1/vaic/skateboard_general_tracking_stu.yaml")
        base = OmegaConf.load(ROOT / "cfg/task/base/hdmi-base.yaml")
        student = OmegaConf.merge(base, original)
        for key in ("observation", "reward", "command", "action", "termination", "randomization", "camera_dr"):
            self.assertEqual(cfg.task[key], student[key], key)
        self.assertEqual(cfg.task.num_envs, 1024)
        self.assertEqual(cfg.algo.cts.teacher_ratio, .5)
        self.assertEqual(cfg.algo.cts.eval_policy, "student")
        self.assertEqual(cfg.algo.runner, "cts")


if __name__ == "__main__":
    unittest.main()
