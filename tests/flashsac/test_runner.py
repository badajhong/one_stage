"""CPU coverage for FlashSAC runner integration without starting Isaac Sim."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import train as runner


class FakeEnv:
    num_envs = 2
    max_episode_length = 4
    current_iter = 10
    device = torch.device("cpu")
    extra = {}
    stats_ema = {}

    def __init__(self, done_schedule=None):
        self.step = 0
        self.progress = []
        self.shared_stat = torch.zeros(self.num_envs, 1)
        self.stat_offset = torch.zeros(self.num_envs, 1)
        self.done_schedule = done_schedule

    def reset(self):
        return TensorDict({
            "policy": torch.zeros(2, 3),
            "depth": torch.zeros(2, 1, 4, 4),
            "belief_hx": torch.zeros(2, 8),
            "is_init": torch.ones(2, 1, dtype=torch.bool),
        }, [2])

    def set_progress(self, iteration):
        self.progress.append(iteration)

    def step_and_maybe_reset(self, td):
        self.step += 1
        self.shared_stat.fill_(self.step)
        self.shared_stat.add_(self.stat_offset)
        done = (
            torch.ones(2, 1, dtype=torch.bool)
            if self.done_schedule is None
            else torch.tensor(self.done_schedule.get(self.step, (False, False))).reshape(2, 1)
        )
        td["next"] = TensorDict({
            "policy": torch.full((2, 3), float(self.step + 100)),
            "depth": torch.full((2, 1, 4, 4), float(self.step + 100)),
            "belief_hx": td["next", "belief_hx"],
            "done": done,
            "terminated": torch.zeros(2, 1, dtype=torch.bool),
            "truncated": torch.ones(2, 1, dtype=torch.bool),
            "reward": torch.ones(2, 1),
            "step_count": torch.full((2, 1), self.step),
            "stats": TensorDict({
                "score": self.shared_stat,
                "tracking_error": -self.shared_stat,
            }, [2]),
        }, [2])
        # The true terminal observation must remain distinct from reset carry.
        return td, self.reset()


class FakePolicy:
    def __init__(self):
        self.transitions = []
        self.events = []

    def get_rollout_policy(self, mode):
        assert mode == "train"

        def rollout(td):
            td["action"] = torch.zeros(2, 1)
            td["next", "belief_hx"] = td["belief_hx"] + 1
            return td

        return rollout

    def add_transition(self, td):
        assert not torch.is_grad_enabled()
        self.events.append("add")
        self.transitions.append(td.clone())

    def update(self):
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        self.events.append("update")

    def pop_info(self):
        return {"update_count": self.events.count("update")}


class RunnerTests(unittest.TestCase):
    def test_environment_count_and_raycast_rendering_config(self):
        for render in (False, True):
            cfg = OmegaConf.create({
                "algo": {"num_envs": 8, "enable_rtx": False},
                "task": {"num_envs": 12, "enable_cameras": True},
                "app": {"enable_cameras": "${task.enable_cameras}"},
                "eval_render": render,
            })
            # Intercept launcher construction before any simulation or W&B side effects.
            with patch.object(runner, "AppLauncher", side_effect=RuntimeError("intercept")) as launch:
                with self.assertRaisesRegex(RuntimeError, "intercept"):
                    runner.main.__wrapped__(cfg)
            self.assertEqual(cfg.task.num_envs, 8)
            self.assertEqual(launch.call_args.args[0]["enable_cameras"], render)

    def test_existing_runner_dispatch_is_preserved(self):
        for name in ("sitt", "cts", "gpo"):
            module = importlib.import_module(f"active_adaptation.learning.{name}.runner")
            cfg = OmegaConf.create({"algo": {"runner": name}})
            with patch.object(module, "train", return_value=name) as train:
                self.assertEqual(runner.main.__wrapped__(cfg), name)
                train.assert_called_once_with(cfg)

    def test_full_observations_stats_and_update_order(self):
        cfg = OmegaConf.create({"total_frames": 8, "algo": {"train_every": 2}})
        env, policy = FakeEnv(), FakePolicy()
        logs, saves = [], []
        run = SimpleNamespace(log=logs.append)
        stats = runner.EpisodeStats([("stats", "score")], device="cpu")
        with patch.object(runner, "tqdm", lambda values, **kwargs: values):
            runner.train_off_policy(
                cfg, env, policy, run, stats,
                lambda p, name: saves.append(name), lambda i: i == 1,
            )
        self.assertEqual(policy.events, ["add", "update"] * 4)
        self.assertEqual(env.progress, [10, 11])
        self.assertEqual(saves, ["checkpoint_1"])
        self.assertEqual(logs[-1]["env_frames"], 8)
        self.assertEqual(logs[0]["train/stats/score"], 1.5)
        for step, transition in enumerate(policy.transitions, 1):
            self.assertTrue(torch.all(transition["next", "depth"] == step + 100))
            self.assertTrue(torch.all(transition["next", "policy"] == step + 100))
            self.assertTrue(torch.all(transition["next", "belief_hx"] == 1))

    def test_legacy_checkpoint_aliases_and_hydra_targets(self):
        for name, config in (
            ("flashsac_vel_flat", "flashsac_vel_flat_train"),
            ("flashsac_vel", "flashsac_vel_train"),
            ("flashsac_vel_finetune", "flashsac_vel_finetune"),
            ("one_stage_flashsac_vel", "one_stage_flashsac_vel"),
        ):
            legacy = importlib.import_module(f"active_adaptation.learning.ppo.{name}")
            current = importlib.import_module(f"active_adaptation.learning.flashsac.{name}")
            self.assertIs(legacy, current)
            target = ConfigStore.instance().load(f"algo/{config}.yaml").node._target_
            self.assertTrue(target.startswith("active_adaptation.learning.flashsac."))

    def run_statistics(self, env, policy, total_frames=4):
        cfg = OmegaConf.create({"total_frames": total_frames, "algo": {"train_every": 2}})
        logs = []
        stats = runner.EpisodeStats([
            ("stats", "score"), ("stats", "tracking_error"),
        ], device="cpu")
        with patch.object(runner, "tqdm", lambda values, **kwargs: values):
            runner.train_off_policy(
                cfg, env, policy, SimpleNamespace(log=logs.append), stats,
                lambda *args: None, lambda i: False,
            )
        return logs

    def test_student_train_and_teacher_metrics_have_matching_components(self):
        for aggregate_mask in (None, [True, False], [False, True], [False, False]):
            with self.subTest(aggregate_mask=aggregate_mask):
                env, policy = FakeEnv(), FakePolicy()
                env.stat_offset[1] = 100
                policy.teacher_env_mask = torch.tensor([True, False])
                policy.student_env_mask = ~policy.teacher_env_mask
                if aggregate_mask is not None:
                    policy.stats_env_mask = torch.tensor(aggregate_mask)
                log = self.run_statistics(env, policy)[0]
                self.assertEqual(log["train/stats/score"], 101.5)
                self.assertEqual(log["teacher/stats/score"], 1.5)
                self.assertEqual(log["train/stats/tracking_error"], -101.5)
                self.assertEqual(log["teacher/stats/tracking_error"], -1.5)
                expected = {
                    "stats/score", "stats/tracking_error",
                    "stats/episode_time_limit", "stats/command_finished",
                }
                self.assertEqual({k.removeprefix("train/") for k in log if k.startswith("train/")}, expected)
                self.assertEqual({k.removeprefix("teacher/") for k in log if k.startswith("teacher/")}, expected)
                self.assertFalse(any(k.startswith(("train/student/", "train/teacher/")) for k in log))

    def test_generic_statistics_keep_optional_aggregate_filter(self):
        for teacher_mask_present in (False, True):
            with self.subTest(teacher_mask_present=teacher_mask_present):
                env, policy = FakeEnv(), FakePolicy()
                env.stat_offset[1] = 100
                policy.stats_env_mask = torch.tensor([False, True])
                if teacher_mask_present:
                    policy.teacher_env_mask = torch.tensor([True, False])
                log = self.run_statistics(env, policy)[0]
                self.assertEqual(log["train/stats/score"], 101.5)
                if teacher_mask_present:
                    self.assertEqual(log["teacher/stats/score"], 1.5)
                else:
                    self.assertFalse(any(k.startswith("teacher/") for k in log))

    def test_role_statistics_accumulate_independently_across_async_episode_ends(self):
        env = FakeEnv(done_schedule={
            1: (True, False), 2: (True, False),
            3: (False, True), 4: (True, False),
            5: (False, True), 7: (True, False),
            11: (False, True), 14: (False, True),
        })
        policy = FakePolicy()
        env.stat_offset[1] = 100
        policy.teacher_env_mask = torch.tensor([True, False])
        policy.student_env_mask = ~policy.teacher_env_mask
        logs = self.run_statistics(env, policy, total_frames=28)
        self.assertEqual(logs[0]["teacher/stats/score"], 1.5)
        self.assertFalse(any(k.startswith("train/") for k in logs[0]))
        self.assertEqual(logs[3]["teacher/stats/score"], 5.5)
        self.assertEqual(logs[3]["train/stats/score"], 104.0)
        self.assertEqual(logs[6]["train/stats/score"], 112.5)
        self.assertFalse(any(k.startswith("teacher/") for k in logs[6]))
        for index in (1, 2, 4, 5):
            self.assertFalse(any(k.startswith(("train/", "teacher/")) for k in logs[index]))


if __name__ == "__main__":
    unittest.main()
