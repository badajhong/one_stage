"""CPU tests of the real VAIC/SITT modules, without an Isaac Sim process.

Run: python -m unittest discover -s tests/sitt -v
"""

import copy
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

from active_adaptation.learning.sitt.algorithm import SITTPPOVEL
from active_adaptation.learning.sitt.config import SITTPPOConfig
from active_adaptation.learning.sitt.losses import action_kl, frozen, ramp_factor, shared_covariance_kl
from active_adaptation.learning.sitt.runner import RolloutCollector
from active_adaptation.learning.ppo.ppo_vel import PPOVEL
from active_adaptation.learning.ppo.common import GAE
from active_adaptation.learning.modules.rnn import set_recurrent_mode


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]


class FakeEnv:
    def __init__(self, n=4, device="cpu"):
        self.num_envs = n
        self.device = torch.device(device)
        self.step_dt = .02
        self.current_iter = 0
        self.cfg = SimpleNamespace(reward={"tracking": {}, "feet": {}})
        self.action_manager = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(6)])
        shapes = {"command": (10,), "policy": (20,), "priv": (24,),
                  "object_": (21,), "object_geo_": (384,), "ref_joint_pos_": (6,),
                  "vel_command": (25,), "depth": (1, 36, 64)}
        self.observation_spec = Composite({
            k: Unbounded((n, *shape), device=self.device) for k, shape in shapes.items()
        }, shape=(n,), device=self.device)
        self.action_spec = Unbounded((n, 6), device=self.device)
        self.reward_spec = Unbounded((n, 2), device=self.device)

    def set_progress(self, value):
        self.current_iter = value


def make_policy(**settings):
    env = FakeEnv()
    cfg = SITTPPOConfig(train_every=8, ppo_epochs=1, num_minibatches=2, desired_kl=None)
    cfg.sitt.warmup_iters = 0
    cfg.sitt.ramp_iters = 0
    cfg.sitt.alignment_epochs = 1
    cfg.sitt.proxy_epochs = 1
    for key, value in settings.items():
        setattr(cfg.sitt, key, value)
    return SITTPPOVEL(cfg, env.observation_spec, env.action_spec, env.reward_spec, env.device, env)


def observation(policy, n=None):
    td = policy.observation_spec.rand()
    n = n or td.shape[0]
    td["is_init"] = torch.zeros(n, 1, dtype=torch.bool)
    td["adapt_hx"] = torch.randn(n, policy.cfg.latent_dim)
    td["depth_hx"] = torch.randn(n, 64)
    td["step_count"] = torch.full((n, 1), 10, dtype=torch.long)
    return td


@torch.no_grad()
def rollout(policy):
    steps = []
    td = observation(policy)
    collector = policy.get_rollout_policy("train")
    for t in range(policy.cfg.train_every):
        with set_exploration_type(ExplorationType.RANDOM):
            td = collector(td)
        td["state_value"] = policy.critic(td.copy())["state_value"]
        td["sitt_value"] = policy.sitt_critic(td.copy())["sitt_value"]
        next_obs = observation(policy)
        next_obs["adapt_hx"] = td["next", "adapt_hx"]
        next_obs["depth_hx"] = td["next", "depth_hx"]
        # Include a reset in the middle of the sequence.
        if t == 3:
            next_obs["is_init"][0] = True
            next_obs["adapt_hx"][0] = 0
            next_obs["depth_hx"][0] = 0
        nxt = TensorDict({
            "reward": torch.rand(td.shape[0], 2),
            "done": next_obs["is_init"].clone(), "terminated": next_obs["is_init"].clone(),
            "discount": torch.ones(td.shape[0], 1),
            "state_value": policy.critic(next_obs.copy())["state_value"],
            "sitt_value": policy.sitt_critic(next_obs.copy())["sitt_value"],
        }, td.batch_size)
        td["next"] = nxt
        steps.append(td.clone())
        td = next_obs
    return torch.stack(steps, 1)


def weights(module):
    return {k: v.detach().clone() for k, v in module.named_parameters()}


def changed(before, module):
    return any(not torch.equal(before[k], v) for k, v in module.named_parameters())


class TestConfiguration(unittest.TestCase):
    def assert_task_parity(self, sitt_task, student_task, teacher_task):
        with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
            cfg = compose(config_name="train", overrides=["algo=sitt_ppo_vel_train", f"task={sitt_task}"])
        base = OmegaConf.load(ROOT / "cfg/task/base/hdmi-base.yaml")
        student = OmegaConf.merge(base, OmegaConf.load(ROOT / f"cfg/task/G1/vaic/{student_task}.yaml"))
        teacher = OmegaConf.merge(base, OmegaConf.load(ROOT / f"cfg/task/G1/vaic/{teacher_task}.yaml"))
        for key in ("observation", "reward", "command", "action", "termination", "randomization", "camera_dr"):
            self.assertEqual(cfg.task[key], student[key], key)
        self.assertEqual(cfg.task.reward, teacher.reward)
        self.assertNotIn("height", cfg.task.observation)
        self.assertEqual(cfg.task.camera_width, student.camera_width)
        self.assertEqual(cfg.task.camera_height, student.camera_height)
        self.assertTrue(cfg.task.enable_cameras)
        self.assertEqual(cfg.algo.runner, "sitt")

    def test_task_parity(self):
        self.assert_task_parity(
            "G1/sitt/skateboard", "skateboard_stu", "skateboard_tea"
        )

    def test_general_tracking_task_parity(self):
        self.assert_task_parity(
            "G1/sitt/skateboard_general_tracking",
            "skateboard_general_tracking_stu",
            "skateboard_general_tracking_tea",
        )


class TestObjectives(unittest.TestCase):
    def test_shared_covariance_kl_value_and_gradients(self):
        loc = torch.tensor([[1., 2.]], requires_grad=True)
        target = torch.tensor([[0., 0.]], requires_grad=True)
        std = torch.tensor([.5, 2.], requires_grad=True)
        loss = shared_covariance_kl(loc, target, std).sum()
        torch.testing.assert_close(loss, action_kl(loc, std, target, std).sum())
        loss.backward()
        torch.testing.assert_close(loc.grad, torch.tensor([[4., .5]]))
        torch.testing.assert_close(std.grad, torch.tensor([-8., -.5]))
        self.assertIsNone(target.grad)
        torch.testing.assert_close(
            shared_covariance_kl(loc, target, std * 2),
            shared_covariance_kl(loc, target, std) / 4,
        )

    def test_kl_direction_and_stop_gradient(self):
        loc = torch.tensor([[1., 2.]], requires_grad=True)
        scale = torch.tensor([[.5, 2.]], requires_grad=True)
        target = torch.zeros_like(loc, requires_grad=True)
        target_scale = torch.ones_like(loc, requires_grad=True)
        result = action_kl(loc, scale, target, target_scale)
        expected = (torch.log(target_scale / scale) + (scale.square() + loc.square()) / 2 - .5).sum()
        torch.testing.assert_close(result.squeeze(), expected)
        result.sum().backward()
        self.assertIsNotNone(loc.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(target_scale.grad)
        self.assertLess((-result).item(), 0)

    def test_schedule(self):
        self.assertEqual(ramp_factor(99, 100, 200), 0)
        self.assertEqual(ramp_factor(200, 100, 200), .5)
        self.assertEqual(ramp_factor(300, 100, 200), 1)
        self.assertEqual(ramp_factor(100, 100, 0), 1)

    def test_split_gae_matches_combined_reward(self):
        gae = GAE(.99, .95)
        rewards, penalty = torch.randn(4, 8, 2), -torch.rand(4, 8, 1)
        values, next_values = torch.randn(4, 8, 2), torch.randn(4, 8, 2)
        pv, npv = torch.randn(4, 8, 1), torch.randn(4, 8, 1)
        done = torch.zeros(4, 8, 1, dtype=torch.bool)
        done[0, 3] = True
        terminated = done.clone()
        terminated[0, 3] = False  # time-limit: bootstrap value but stop trace
        discount = torch.rand(4, 8, 1)
        a, _ = gae(rewards, terminated, done, values, next_values, discount)
        pa, _ = gae(penalty, terminated, done, pv, npv, discount)
        coefficient = .04
        combined, _ = gae(rewards.mean(-1, keepdim=True) + coefficient * penalty,
                          terminated, done, values.mean(-1, keepdim=True) + coefficient * pv,
                          next_values.mean(-1, keepdim=True) + coefficient * npv, discount)
        torch.testing.assert_close(a.mean(-1, keepdim=True) + coefficient * pa, combined)


class TestSITT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.policy = make_policy()

    def test_shared_decoder_and_student_observation_boundary(self):
        p = self.policy
        obs = observation(p)
        with torch.no_grad():
            teacher = p.encode_teacher(obs.copy())
            teacher["priv_pred"] = teacher["priv_feature"]
            expected = p.shared_actor.get_dist(teacher)
            teacher_mean = expected.mean.clone()
            with set_exploration_type(ExplorationType.MODE):
                collected = p.get_rollout_policy("train")(obs.copy())
            torch.testing.assert_close(collected["action"], teacher_mean)
            torch.testing.assert_close(collected["loc"], teacher_mean)
            torch.testing.assert_close(collected["sample_log_prob"], expected.log_prob(teacher_mean))
            a = p.shared_actor.get_dist(p.encode_student(obs.copy())).mean
            corrupted = obs.clone()
            for key in ("priv", "object_", "command", "ref_joint_pos_"):
                corrupted[key].fill_(123.)
            b = p.shared_actor.get_dist(p.encode_student(corrupted)).mean
            torch.testing.assert_close(a, b)
            # The shared decoder has the student-safe input contract and no
            # privileged reference-joint residual.
            shifted = obs.clone()
            shifted["ref_joint_pos_"] += 2
            shifted_teacher = p.encode_teacher(shifted)
            shifted_teacher["priv_pred"] = shifted_teacher["priv_feature"]
            shift_dist = p.shared_actor.get_dist(shifted_teacher)
            torch.testing.assert_close(shift_dist.mean, teacher_mean)
        self.assertFalse(hasattr(p, "actor"))
        self.assertFalse(hasattr(p, "actor_adapt"))
        self.assertEqual(
            p.shared_actor.module[0][0].in_keys,
            ["vel_command", "policy", "priv_pred"],
        )
        self.assertIn("object_pred", p.adapt_module[0].in_keys)
        self.assertNotIn("object_", p.adapt_module[0].in_keys)

    def test_recurrent_sequence_matches_step_rollout_with_reset(self):
        p = self.policy
        batch = rollout(p)
        with torch.no_grad():
            sequential = []
            hx, dhx = batch["adapt_hx"][:, 0], batch["depth_hx"][:, 0]
            for t in range(batch.shape[1]):
                obs = batch[:, t].copy()
                obs["adapt_hx"], obs["depth_hx"] = hx, dhx
                result = p.encode_student(obs)
                sequential.append(result["priv_pred"])
                hx, dhx = result["next", "adapt_hx"], result["next", "depth_hx"]
            with set_recurrent_mode(True):
                whole = p.encode_student(batch.copy())
            torch.testing.assert_close(torch.stack(sequential, 1), whole["priv_pred"], rtol=1e-5, atol=1e-5)

    def test_gradient_and_optimizer_isolation(self):
        p = self.policy
        batch = rollout(p)
        reward_before = batch["next", "reward"].clone()
        decoder_before = weights(p.shared_actor)
        teacher_before = weights(p.encoder_priv)
        student_before = weights(p.adapt_module)
        proxy_before = weights(p.proxy_student)
        p.train_policy(batch.copy())
        self.assertTrue(changed(decoder_before, p.shared_actor))
        self.assertTrue(changed(teacher_before, p.encoder_priv))
        self.assertFalse(changed(student_before, p.adapt_module))
        self.assertFalse(changed(proxy_before, p.proxy_student))
        decoder_before = weights(p.shared_actor)
        teacher_before = weights(p.encoder_priv)
        p.align_student(batch)
        self.assertFalse(changed(decoder_before, p.shared_actor))
        self.assertFalse(changed(teacher_before, p.encoder_priv))
        self.assertTrue(changed(student_before, p.adapt_module))
        self.assertFalse(changed(proxy_before, p.proxy_student))
        decoder_before = weights(p.shared_actor)
        encoder_before = weights(p.adapt_module)
        p.align_proxy(batch)
        self.assertTrue(changed(proxy_before, p.proxy_student))
        self.assertFalse(changed(decoder_before, p.shared_actor))
        self.assertFalse(changed(encoder_before, p.adapt_module))
        torch.testing.assert_close(reward_before, batch["next", "reward"])

    def test_direct_kl_updates_teacher_encoder_and_std_not_mean_decoder(self):
        p = self.policy
        q = make_policy()
        q.load_state_dict(copy.deepcopy(p.state_dict()))
        p.cfg.sitt.reward_kl_coef = 0
        q.cfg.sitt.reward_kl_coef = 0
        p.cfg.sitt.policy_kl_coef = 0
        q.cfg.sitt.policy_kl_coef = 1
        # Isolate the loss gradients from joint norm clipping across mean/std.
        p.cfg.max_grad_norm = q.cfg.max_grad_norm = 1e6
        batch = rollout(p)
        zero_kl, task_only = batch.clone(), batch.clone()
        p._compute_advantage(zero_kl, p.critic)
        PPOVEL._compute_advantage(p, task_only, p.critic)
        torch.testing.assert_close(zero_kl["adv"], task_only["adv"], rtol=0, atol=0)
        torch.testing.assert_close(zero_kl["ret"], task_only["ret"], rtol=0, atol=0)
        q_batch = zero_kl.clone()
        p._update_ppo(zero_kl.reshape(-1))
        q._update_ppo(q_batch.reshape(-1))
        for (name, x), y in zip(p.shared_actor.named_parameters(), q.shared_actor.parameters()):
            if not name.endswith("actor_std"):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue(any(
            not torch.equal(x, y)
            for x, y in zip(p.encoder_priv.parameters(), q.encoder_priv.parameters())
        ))
        q.zero_grad(set_to_none=True)
        q.direct_policy_kl(batch.reshape(-1)).mean().backward()
        self.assertTrue((q.action_std.grad < 0).any())
        self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0
                            for x in q.encoder_priv.parameters()))
        for name, param in q.shared_actor.named_parameters():
            if not name.endswith("actor_std"):
                self.assertIsNone(param.grad)
        for module in (q.proxy_student, q.adapt_module, q.object_adapt, q.temporal_depth_gru):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_direct_kl_recomputes_current_proxy_and_ignores_rollout_cache(self):
        p = self.policy
        batch = rollout(p)
        p._compute_advantage(batch, p.critic)
        flat = batch.reshape(-1)
        with torch.no_grad():
            before = p.direct_policy_kl(flat).clone()
            p.shared_actor.module[0][2].module.actor_mean.bias.add_(.25)
            torch.testing.assert_close(p.direct_policy_kl(flat), before, atol=1e-8, rtol=1e-5)
            p.action_std.mul_(2.)
            torch.testing.assert_close(p.direct_policy_kl(flat), before / 4, atol=1e-8, rtol=1e-5)
            proxy = p.proxy_distribution(flat.copy())
            teacher = p.encode_teacher(flat.copy())
            teacher['priv_pred'] = teacher['priv_feature']
            teacher_dist = p.shared_actor.get_dist(teacher)
            # Generic Gaussian KL subtracts nearly equal float32 terms for
            # small mismatches; use float64 for this independent reference.
            expected = action_kl(teacher_dist.mean.double(), teacher_dist.scale.double(),
                                 proxy.mean.double(), proxy.scale.double()).float()
            torch.testing.assert_close(p.direct_policy_kl(flat), expected, atol=1e-7, rtol=1e-5)
        # Historical proxy distributions must never enter the direct PPO loss.
        flat['proxy_loc'].fill_(float('nan'))
        flat['proxy_scale'].fill_(float('nan'))
        info = p._update_ppo(flat)
        torch.testing.assert_close(info['sitt/teacher_proxy_kl'], expected.mean(), atol=1e-7, rtol=1e-5)

    def test_action_alignment_reaches_all_student_perception_modules(self):
        p = self.policy
        batch = rollout(p)
        p.zero_grad(set_to_none=True)
        with set_recurrent_mode(True), frozen(p.shared_actor):
            with torch.no_grad():
                teacher = p.encode_teacher(batch.copy())
                teacher['priv_pred'] = teacher['priv_feature']
                target_mean = p.shared_actor.get_dist(teacher).mean
            student = p.encode_student(batch.copy())
            (p.shared_actor.get_dist(student).mean - target_mean).abs().mean().backward()
        for module in (p.depth_cnn, p.temporal_depth_gru.gru, p.object_adapt, p.adapt_module):
            self.assertGreater(sum(x.grad.abs().sum().item() for x in module.parameters()
                                   if x.grad is not None), 0.)
        for module in (p.shared_actor, p.encoder_priv, p.proxy_student):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_paired_alignment_order_and_current_student_target(self):
        p = self.policy
        batch = rollout(p)
        events = []
        student_step, proxy_step = p._student_alignment_step, p._proxy_alignment_step
        def student(b):
            events.append(('student', b))
            return student_step(b)
        def proxy(b):
            self.assertEqual(events[-1][0], 'student')
            self.assertIs(events[-1][1], b)
            with torch.no_grad():
                target = p.encode_student(b.copy())['priv_pred'].clone()
                feature = p.encode_proxy(b.copy())['proxy_feature']
                expected = (feature - target).abs().mean().item()
            events.append(('proxy', b))
            result = proxy_step(b)
            self.assertAlmostEqual(result['proxy/feature_loss'], expected, places=6)
            return result
        with patch.object(p, '_student_alignment_step', side_effect=student), \
                patch.object(p, '_proxy_alignment_step', side_effect=proxy):
            p.align_encoders(batch)
        self.assertEqual([x[0] for x in events], ['student', 'proxy'] * p.cfg.num_minibatches)

    def test_proxy_uses_teacher_observations_not_perception_modules(self):
        p = self.policy
        obs = observation(p)
        self.assertEqual(p.proxy_student.in_keys,
                         ['priv', 'object_', 'object_trans'])
        teacher_linears = [x for x in p.encoder_priv[1].module.modules()
                           if isinstance(x, torch.nn.Linear)]
        proxy_linears = [x for x in p.proxy_student.net.modules()
                         if isinstance(x, torch.nn.Linear)]
        self.assertEqual([(x.in_features, x.out_features) for x in proxy_linears],
                         [(x.in_features, x.out_features) for x in teacher_linears])
        with torch.no_grad():
            expected = p.encode_proxy(obs.copy())['proxy_feature']
            corrupted = obs.clone()
            for key in ('depth', 'adapt_hx', 'depth_hx', 'command', 'policy',
                        'vel_command', 'ref_joint_pos_'):
                corrupted[key].fill_(float('nan'))
            torch.testing.assert_close(p.encode_proxy(corrupted)['proxy_feature'], expected)
        proxy_ids = {id(x) for x in p.proxy_student.parameters()}
        for module in (p.encoder_priv, p.temporal_depth_gru, p.object_adapt, p.adapt_module, p.shared_actor):
            self.assertFalse(proxy_ids & {id(x) for x in module.parameters()})

    def test_checkpoint_does_not_mutate_and_restores_optimizer(self):
        p = self.policy
        info = p.train_op(rollout(p))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in info.values()))
        before = weights(p)
        state = copy.deepcopy(p.state_dict())
        self.assertEqual(state['sitt_objective_version'], 2)
        self.assertEqual(state['sitt_version'], 3)
        self.assertFalse(changed(before, p))
        restored = make_policy()
        restored.load_state_dict(state)
        self.assertEqual(restored.num_updates, 1)
        for key, value in restored.named_parameters():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for name in p.optimizer_names:
            original = getattr(p, name).state_dict()["state"]
            loaded = getattr(restored, name).state_dict()["state"]
            self.assertEqual(original.keys(), loaded.keys())
            for key in original:
                for field in original[key]:
                    torch.testing.assert_close(original[key][field], loaded[key][field])
        with self.assertRaisesRegex(ValueError, "SITT checkpoint"):
            restored.load_state_dict({"last_phase": "train"})
        # Same next data + minibatch RNG must produce the same resumed update.
        next_batch = rollout(p)
        torch.manual_seed(123)
        p.train_op(next_batch.clone())
        torch.manual_seed(123)
        restored.train_op(next_batch.clone())
        for a, b in zip(p.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_collector_preserves_terminal_values_and_histories(self):
        p = self.policy
        env = p.env
        recorded = []
        env.reset = lambda: observation(p)

        def step(carry):
            nxt = observation(p)
            done = torch.zeros(env.num_envs, 1, dtype=torch.bool)
            done[0] = True
            nxt.update({"reward": torch.ones(env.num_envs, 2), "done": done,
                        "terminated": torch.zeros_like(done), "discount": torch.ones(env.num_envs, 1)})
            nxt["adapt_hx"] = carry["next", "adapt_hx"]
            nxt["depth_hx"] = carry["next", "depth_hx"]
            td = carry.copy()
            td["next"] = nxt
            recorded.append(p.critic(nxt.copy())["state_value"].clone())
            reset = nxt.clone()
            for key in env.observation_spec.keys():
                reset[key][0].zero_()
            reset["adapt_hx"][0].zero_()
            reset["depth_hx"][0].zero_()
            reset["is_init"][0] = True
            return td, reset

        env.step_and_maybe_reset = step
        collector = RolloutCollector(env, p)
        data = collector.collect()
        torch.testing.assert_close(data["next", "state_value"], torch.stack(recorded, 1))
        self.assertEqual(data["depth_hx"].shape, (4, 8, 64))
        self.assertEqual(data["adapt_hx"].shape, (4, 8, 256))
        self.assertTrue(torch.equal(data["next", "reward"], torch.ones(4, 8, 2)))
        # Reusing the collector must not depend on stale buffer keys or live tensors.
        saved = data.clone()
        collector.collect()
        torch.testing.assert_close(saved["next", "state_value"], torch.stack(recorded[:8], 1))


if __name__ == "__main__":
    unittest.main()
