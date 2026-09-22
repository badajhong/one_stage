"""Guider-only trajectories; learner-only final evaluation."""

import datetime
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType


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
                # Value depends only on guider state. Do not advance the learner
                # history twice; bootstrap the actual next observation before reset.
                nxt = self.policy.encode_guider(td["next"].copy())
                self.policy.critic(nxt)
                transition = td.select(
                    *self.observation_keys, "is_init", "step_count", "adapt_hx", "depth_hx",
                    "action", "sample_log_prob", "loc", "scale", "gpo_latent", "state_value",
                )
                transition["next"] = nxt.select(
                    "reward", "done", "terminated", "discount", "stats", "state_value", strict=False,
                )
                if self.buffer is None:
                    n, t = self.env.num_envs, self.policy.cfg.train_every
                    self.buffer = TensorDict({}, [n, t], device=self.env.device)
                    for key, value in transition.items(True, True):
                        self.buffer[key] = torch.empty(n, t, *value.shape[1:], dtype=value.dtype, device=value.device)
                    logging.info("GPO rollout buffer: %.1f MB", self.buffer.bytes() / 1e6)
                self.buffer[:, step] = transition
        return self.buffer


class EpisodeMetrics:
    def __init__(self):
        self.sums = {}
        self.episodes = 0

    def add(self, data):
        nxt = data["next"]
        done = nxt["done"].squeeze(-1)
        count = done.sum().item()
        if count:
            for key, value in nxt["stats"].items(True, True):
                name = "/".join(key) if isinstance(key, tuple) else key
                self.sums[name] = self.sums.get(name, 0.) + value[done].float().sum().item()
            self.episodes += count

    def pop(self):
        if not self.episodes:
            return {}
        info = {f"train/stats/{k}": v / self.episodes for k, v in self.sums.items()}
        self.sums.clear()
        self.episodes = 0
        return info


@torch.no_grad()
def evaluate_learner(env, policy, seed=0):
    """Each env's first completed episode, with the observation normalizer frozen."""
    env.eval()
    env.base_env.eval()
    env.set_seed(seed)
    rollout = policy.get_rollout_policy("learner")
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
    info = {}
    for key, chunks in samples.items():
        values = torch.cat(chunks)
        info[f"eval/{key}"] = values.mean().item()
        info[f"eval/{key}_std"] = values.std().item() if values.numel() > 1 else 0.
    info["episode_cnt"] = completed.sum().item()
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


def validate_resume(saved, cfg):
    from omegaconf import OmegaConf

    def settings(algo):
        result = OmegaConf.to_container(algo, resolve=True)
        result["gpo"] = {k: v for k, v in result["gpo"].items()
                         if k not in ("eval_policy", "evaluate_at_end")}
        return result

    if (saved.algo.get("runner") != "gpo" or saved.task != cfg.task
            or saved.vecnorm != cfg.vecnorm or settings(saved.algo) != settings(cfg.algo)):
        raise ValueError("GPO training resume requires the saved task and algorithm settings; "
                         "only eval_policy/evaluate_at_end and run-level settings may change.")


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
        raise ValueError("GPO currently supports a single process/GPU.")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if (cfg.algo.num_minibatches < 1 or cfg.task.num_envs < cfg.algo.num_minibatches
            or cfg.task.num_envs % cfg.algo.num_minibatches):
        raise ValueError("GPO num_envs must be divisible by num_minibatches.")
    if cfg.checkpoint_path and not cfg.checkpoint_path.startswith("run:"):
        cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    # Resolve/validate before starting the simulator; saved task is the filtered
    # configuration used by make_env_policy, whose filtering is applied here too.
    for key in list(cfg.task.observation.keys()):
        if key not in cfg.algo.in_keys and not key.endswith("_"):
            cfg.task.observation.pop(key)
    resume_rng = None
    if cfg.checkpoint_path:
        cfg.checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
        resume = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        validate_resume(resume["cfg"], cfg)
        resume_rng = resume.get("rng")
        del resume
    app = AppLauncher(OmegaConf.to_container(cfg.app), device="cuda:0").app
    env = run = None
    try:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
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
            print(f"[GPO] Saved checkpoint: {path}", flush=True)

        frames = env.num_envs * cfg.algo.train_every
        total_iters = cfg.total_frames // frames
        if total_iters <= policy.num_updates:
            raise ValueError("total_frames must exceed the resumed GPO frame count.")
        if resume_rng:
            torch.set_rng_state(resume_rng["torch"])
            np.random.set_state(resume_rng["numpy"])
            random.setstate(resume_rng["python"])
            if resume_rng["cuda"]:
                torch.cuda.set_rng_state_all(resume_rng["cuda"])
        collector = RolloutCollector(env, policy)
        metrics = EpisodeMetrics()
        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        print(f"[GPO-{cfg.algo.gpo.variant}] {env.num_envs} guider envs; "
              "learner perception jointly trained; final eval = learner.", flush=True)
        for i in tqdm(range(policy.num_updates, total_iters)):
            env.set_progress(policy.num_updates)
            start = time.perf_counter()
            data = collector.collect()
            rollout_seconds = time.perf_counter() - start
            metrics.add(data)
            info = {}
            if i % log_interval == 0 or i == total_iters - 1:
                info.update(metrics.pop())
            start = time.perf_counter()
            info.update(policy.train_op(data))
            info["training_time"] = time.perf_counter() - start
            info["rollout_fps"] = frames / rollout_seconds
            info["env_frames"] = policy.num_updates * frames
            # Same dashboard schema as scripts/train.py / ppo_vel_train.
            # All training performance metrics describe guider trajectories.
            info.update(env.extra)
            info.update(env.stats_ema)
            run.log(info)
            if cfg.save_interval > 0 and policy.num_updates % cfg.save_interval == 0:
                save(f"checkpoint_{policy.num_updates}")
        env.set_progress(policy.num_updates)
        save("checkpoint_final")
        if cfg.algo.gpo.evaluate_at_end:
            evaluation = evaluate_learner(env, policy, cfg.seed)
            run.log(evaluation)
            print(f"[GPO] Learner evaluation: {evaluation['episode_cnt']} completed, "
                  f"{env.num_envs - evaluation['episode_cnt']} incomplete, "
                  f"success={evaluation.get('eval/success', float('nan')):.4f}", flush=True)
        return f"{run.entity}/{run.project}/{run.id}"
    finally:
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
