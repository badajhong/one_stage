"""SITT-only collection, checkpointing and training entry point.

Selected by scripts/train.py only when algo.runner=sitt. The original runner,
environment, reward manager and other algorithms retain their previous behavior.
"""

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
    """Store synchronized observations and true next-state bootstrap values."""

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.rollout_policy = policy.get_rollout_policy("train")
        self.carry = env.reset()
        self.buffer = None
        self.observation_keys = list(env.observation_spec.keys())

    @torch.no_grad()
    def collect(self):
        with set_exploration_type(ExplorationType.RANDOM):
            for step in range(self.policy.cfg.train_every):
                carry = self.rollout_policy(self.carry)
                td, self.carry = self.env.step_and_maybe_reset(carry)
                # td.next still contains the terminal observation; carry may
                # already contain a reset observation. Bootstrap from td.next.
                td["state_value"] = self.policy.critic(td.copy())["state_value"]
                td["sitt_value"] = self.policy.sitt_critic(td.copy())["sitt_value"]
                nxt = td["next"].copy()
                nxt["state_value"] = self.policy.critic(nxt.copy())["state_value"]
                nxt["sitt_value"] = self.policy.sitt_critic(nxt.copy())["sitt_value"]
                transition = td.select(
                    *self.observation_keys, "is_init", "step_count", "adapt_hx", "depth_hx",
                    "action", "sample_log_prob", "loc", "scale", "proxy_loc", "proxy_scale",
                    "sitt_kl", "state_value", "sitt_value",
                )
                transition["next"] = nxt.select(
                    "reward", "done", "terminated", "discount", "stats", "state_value", "sitt_value",
                    strict=False,
                )
                if self.buffer is None:
                    n, t = self.env.num_envs, self.policy.cfg.train_every
                    self.buffer = TensorDict({}, [n, t], device=self.env.device)
                    for key, value in transition.items(True, True):
                        self.buffer[key] = torch.empty(
                            n, t, *value.shape[1:], dtype=value.dtype, device=value.device
                        )
                    logging.info("SITT rollout buffer: %.1f MB", self.buffer.bytes() / 1e6)
                # copy into owned storage before the simulator reuses tensors
                self.buffer[:, step] = transition
        return self.buffer


def checkpoint(policy, env, vecnorm, cfg, run):
    """Full learning-state checkpoint; a resume starts fresh simulator episodes."""
    return {
        "policy": policy.state_dict(), "env": env.state_dict(),
        "vecnorm": vecnorm.state_dict(), "cfg": cfg,
        "wandb": {"id": run.id, "name": run.name},
        "rng": {
            "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def train(cfg):
    # Imports remain lazy: unit tests and Hydra composition need no simulator.
    import wandb
    from hydra.utils import to_absolute_path
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import active_adaptation as aa
    from helpers import make_env_policy, EpisodeStats, evaluate
    from active_adaptation.utils.wandb import parse_checkpoint_path

    if aa.is_distributed():
        raise ValueError("SITT currently supports one process/GPU; distributed gradient synchronization is not implemented.")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.checkpoint_path and not cfg.checkpoint_path.startswith("run:"):
        cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    app = AppLauncher(OmegaConf.to_container(cfg.app), device="cuda:0").app
    env = None
    run = None
    try:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        resume_rng = None
        if cfg.checkpoint_path:
            cfg.checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
            resume = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
            resume_rng = resume.get("rng")
            saved_cfg = resume["cfg"]
            if (saved_cfg.task.num_envs != cfg.task.num_envs
                    or saved_cfg.algo.train_every != cfg.algo.train_every):
                raise ValueError("SITT training resume requires the saved num_envs and train_every (frame accounting).")
            del resume
        env, policy, vecnorm = make_env_policy(cfg)
        run = wandb.init(project=cfg.wandb.project, job_type=cfg.wandb.job_type,
                         mode=cfg.wandb.mode, tags=cfg.wandb.tags,
                         name=f"{cfg.exp_name}-{datetime.datetime.now():%Y-%m-%d-%H-%M}")
        # Disabled W&B runs do not create run.dir themselves.
        Path(run.dir).mkdir(parents=True, exist_ok=True)
        run.config.update(OmegaConf.to_container(cfg))
        # cfg.task.reward is mutated internally by the original env; retain all
        # effective observation/reward terms in the checkpoint config.
        cfg_path = Path(run.dir) / "cfg.yaml"
        OmegaConf.save(cfg, cfg_path)
        run.save(str(cfg_path), policy="now")
        package_dir = Path(__file__).parent
        for source in package_dir.glob("*.py"):
            # W&B stores all SITT sources, rather than only the algorithm class.
            run.save(str(source), base_path=str(package_dir), policy="now")

        def save(name):
            path = Path(run.dir) / f"{name}.pt"
            torch.save(checkpoint(policy, env, vecnorm, cfg, run), path)
            run.save(str(path), base_path=run.dir, policy="now")
            logging.info("Saved SITT checkpoint: %s", path)
            print(f"[SITT] Saved checkpoint: {path}", flush=True)

        batch_frames = env.num_envs * cfg.algo.train_every
        total_iters = cfg.total_frames // batch_frames
        if total_iters <= policy.num_updates:
            raise ValueError("total_frames must exceed the resumed checkpoint's completed frame budget.")
        if resume_rng is not None:
            torch.set_rng_state(resume_rng["torch"])
            np.random.set_state(resume_rng["numpy"])
            random.setstate(resume_rng["python"])
            if resume_rng["cuda"]:
                torch.cuda.set_rng_state_all(resume_rng["cuda"])
        collector = RolloutCollector(env, policy)
        stats_keys = [k for k in env.reward_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"]
        episode_stats = EpisodeStats(stats_keys, device=env.device)
        log_interval = env.max_episode_length // cfg.algo.train_every + 1
        for _ in tqdm(range(policy.num_updates, total_iters)):
            env.set_progress(policy.num_updates)
            start = time.perf_counter()
            data = collector.collect()
            rollout_time = time.perf_counter() - start
            episode_stats.add(data)
            start = time.perf_counter()
            info = policy.train_op(data)
            info["training_time"] = time.perf_counter() - start
            info["rollout_fps"] = batch_frames / rollout_time
            info["env_frames"] = policy.num_updates * batch_frames
            info["iterations"] = policy.num_updates
            if policy.num_updates % log_interval == 0 and len(episode_stats):
                for key, value in episode_stats.pop().items(True, True):
                    info["train/" + ("/".join(key) if isinstance(key, tuple) else key)] = value.float().mean().item()
            info.update(env.extra)
            info.update(env.stats_ema)
            run.log(info)
            if cfg.save_interval > 0 and policy.num_updates % cfg.save_interval == 0:
                save(f"checkpoint_{policy.num_updates}")
        env.set_progress(policy.num_updates)
        save("checkpoint_final")
        if cfg.algo.sitt.evaluate_at_end:
            for role in ("teacher", "student"):
                info, _, _, _ = evaluate(env, policy.get_rollout_policy(role),
                                          seed=cfg.seed, render=cfg.eval_render)
                run.log({f"{role}/{key}": value for key, value in info.items()})
        return f"{run.entity}/{run.project}/{run.id}"
    except Exception:
        logging.exception("SITT training failed")
        raise
    finally:
        if run is not None:
            run.finish()
        try:
            if env is not None:
                env.close()
        finally:
            # make_env_policy can fail after constructing SimulationContext but
            # before returning env. Unsubscribe its stop callback in that case
            # too; otherwise this Isaac Lab version keeps rendering on shutdown.
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None:
                sim.clear_all_callbacks()
                sim.clear_instance()
            app.close()
