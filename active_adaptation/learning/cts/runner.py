"""CTS collection: VAIC performance keys are student-only; teacher is namespaced."""

import datetime
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType

from .metrics import EpisodeMetrics, RewardMetrics


class RolloutCollector:
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.rollout_policy = policy.get_rollout_policy("train")
        self.carry = env.reset()
        self.buffer = None
        self.observation_keys = list(env.observation_spec.keys())

    @torch.no_grad()
    def collect(self):
        with set_exploration_type(ExplorationType.RANDOM):
            for step in range(self.policy.cfg.train_every):
                carry = self.rollout_policy(self.carry)
                self.policy.critic(carry)
                td, self.carry = self.env.step_and_maybe_reset(carry)
                # Bootstrap the actual terminal next observation, never the
                # auto-reset carry. Continue the history produced by this action.
                nxt = td["next"].copy()
                nxt["is_init"] = torch.zeros_like(td["is_init"])
                nxt = self.policy.encode_role(nxt, "train")
                self.policy.critic(nxt)
                transition = td.select(
                    *self.observation_keys, "is_init", "step_count", "adapt_hx", "depth_hx",
                    "action", "sample_log_prob", "loc", "scale", "cts_latent", "state_value",
                )
                transition["next"] = nxt.select(
                    "reward", "done", "terminated", "discount", "stats", "state_value", strict=False,
                )
                if self.buffer is None:
                    n, t = self.env.num_envs, self.policy.cfg.train_every
                    self.buffer = TensorDict({}, [n, t], device=self.env.device)
                    for key, value in transition.items(True, True):
                        self.buffer[key] = torch.empty(n, t, *value.shape[1:], dtype=value.dtype, device=value.device)
                    logging.info("CTS rollout buffer: %.1f MB", self.buffer.bytes() / 1e6)
                self.buffer[:, step] = transition
        return self.buffer


@torch.no_grad()
def evaluate_student(env, policy, seed=0):
    """One completed episode per env, all using the deployable online student.

    Use VAIC eval keys and reductions, but never select an unfinished episode
    through argmax-on-all-false. Report incomplete episodes explicitly.
    """
    env.eval()
    env.base_env.eval()
    env.set_seed(seed)
    rollout = policy.get_rollout_policy("student")
    carry = env.reset()
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    samples = {}
    with set_exploration_type(ExplorationType.MODE):
        for _ in range(env.max_episode_length + 1):
            td, carry = env.step_and_maybe_reset(rollout(carry))
            first_done = td["next", "done"].squeeze(-1) & ~completed
            if first_done.any():
                for key, value in td["next", "stats"].items(True, True):
                    name = "/".join(key) if isinstance(key, tuple) else key
                    value = value[first_done].float()
                    if key not in ("success", "episode_len"):
                        value = value / td["next", "stats", "episode_len"][first_done].float()
                    samples.setdefault(name, []).append(value.clone())
                completed |= first_done
            if completed.all():
                break
    n = completed.sum().item()
    info = {}
    for key, chunks in samples.items():
        values = torch.cat(chunks)
        info[f"eval/{key}"] = values.mean().item()
        info[f"eval/{key}_std"] = values.std().item() if values.numel() > 1 else 0.
    info["episode_cnt"] = n
    info["eval/incomplete_episodes"] = env.num_envs - n
    return info


def checkpoint(policy, env, vecnorm, cfg, run):
    return {
        "policy": policy.state_dict(), "env": env.state_dict(),
        "vecnorm": vecnorm.state_dict(), "cfg": cfg,
        "wandb": {"id": run.id, "name": run.name},
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }


def train(cfg):
    import wandb
    from hydra.utils import to_absolute_path
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import active_adaptation as aa
    from helpers import make_env_policy
    from active_adaptation.utils.wandb import parse_checkpoint_path

    if aa.is_distributed():
        raise ValueError("CTS currently supports a single process/GPU.")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    nt = int(cfg.task.num_envs * cfg.algo.cts.teacher_ratio)
    for n in (nt, cfg.task.num_envs - nt):
        if cfg.algo.num_minibatches < 1 or n < cfg.algo.num_minibatches or n % cfg.algo.num_minibatches:
            raise ValueError("Each CTS rollout group must be nonempty and divisible by num_minibatches.")
    if cfg.checkpoint_path and not cfg.checkpoint_path.startswith("run:"):
        cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    app = AppLauncher(OmegaConf.to_container(cfg.app), device="cuda:0").app
    env = run = reward_metrics = None
    try:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        resume_rng = None
        if cfg.checkpoint_path:
            cfg.checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
            resume = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
            saved = resume["cfg"]
            training_settings = lambda c: {k: v for k, v in c.items()
                                           if k not in ("eval_policy", "evaluate_at_end")}
            if (saved.algo.get("runner") != "cts" or saved.task.num_envs != cfg.task.num_envs
                    or saved.algo.train_every != cfg.algo.train_every
                    or training_settings(saved.algo.cts) != training_settings(cfg.algo.cts)):
                raise ValueError("CTS training resume requires saved num_envs, train_every and CTS settings.")
            resume_rng = resume.get("rng")
            del resume
        env, policy, vecnorm = make_env_policy(cfg)
        run = wandb.init(project=cfg.wandb.project, job_type=cfg.wandb.job_type,
                         mode=cfg.wandb.mode, tags=cfg.wandb.tags,
                         name=f"{cfg.exp_name}-{datetime.datetime.now():%Y-%m-%d-%H-%M}")
        Path(run.dir).mkdir(parents=True, exist_ok=True)
        run.config.update(OmegaConf.to_container(cfg))
        cfg_path = Path(run.dir) / "cfg.yaml"
        OmegaConf.save(cfg, cfg_path)
        run.save(str(cfg_path), policy="now")
        package_dir = Path(__file__).parent
        for source in package_dir.glob("*.py"):
            run.save(str(source), base_path=str(package_dir), policy="now")

        def save(name):
            path = Path(run.dir) / f"{name}.pt"
            torch.save(checkpoint(policy, env, vecnorm, cfg, run), path)
            run.save(str(path), base_path=run.dir, policy="now")
            print(f"[CTS] Saved checkpoint: {path}", flush=True)

        frames = env.num_envs * cfg.algo.train_every
        total_iters = cfg.total_frames // frames
        if total_iters <= policy.num_updates:
            raise ValueError("total_frames must exceed the resumed CTS frame count.")
        if resume_rng:
            torch.set_rng_state(resume_rng["torch"])
            np.random.set_state(resume_rng["numpy"])
            random.setstate(resume_rng["python"])
            if resume_rng["cuda"]:
                torch.cuda.set_rng_state_all(resume_rng["cuda"])
        reward_metrics = RewardMetrics(env.base_env, policy.num_teacher)
        collector = RolloutCollector(env, policy)
        student_metrics = EpisodeMetrics(slice(policy.num_teacher, None), "train/stats")
        teacher_metrics = EpisodeMetrics(slice(None, policy.num_teacher), "cts_teacher/episode")
        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        print(f"[CTS] {policy.num_teacher} teacher / {policy.num_student} student envs; "
              "W&B: original VAIC keys = student; cts_teacher/* = teacher.", flush=True)
        for i in tqdm(range(policy.num_updates, total_iters)):
            env.set_progress(policy.num_updates)
            start = time.perf_counter()
            data = collector.collect()
            rollout_seconds = time.perf_counter() - start
            student_metrics.add(data)
            teacher_metrics.add(data)
            info = reward_metrics.snapshot()
            if i % log_interval == 0 or i == total_iters - 1:
                info.update(student_metrics.pop())
                info.update(teacher_metrics.pop())
            start = time.perf_counter()
            info.update(policy.train_op(data))
            info["training_time"] = time.perf_counter() - start
            info["rollout_fps"] = frames / rollout_seconds
            info["env_frames"] = policy.num_updates * frames
            info["iterations"] = policy.num_updates
            # Simulator timings describe the full job, not policy performance.
            # Never forward global reward EMAs or aggregated env.extra metrics.
            info.update({k: v for k, v in env.stats_ema.items()
                         if k.startswith(("performance/", "performance_reward/", "performance_update/"))})
            run.log(info)
            if cfg.save_interval > 0 and policy.num_updates % cfg.save_interval == 0:
                save(f"checkpoint_{policy.num_updates}")
        env.set_progress(policy.num_updates)
        reward_metrics.close()
        save("checkpoint_final")
        if cfg.algo.cts.evaluate_at_end:
            # env.eval() freezes VecNorm; every environment is student-driven.
            run.log(evaluate_student(env, policy, cfg.seed))
        return f"{run.entity}/{run.project}/{run.id}"
    finally:
        if reward_metrics is not None:
            reward_metrics.close()
        if run is not None:
            run.finish()
        try:
            if env is not None:
                env.close()
        finally:
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None:
                sim.clear_all_callbacks()
                sim.clear_instance()
            app.close()
