import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import sys
import time
import datetime

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

# Prefer this checkout over an editable install from another VAIC repository.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import active_adaptation as aa
from isaaclab.app import AppLauncher
# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict

# local import
from helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

def train_off_policy(cfg, env, policy, run, episode_stats, save, should_save):
    """Off-policy loop (FlashSAC): every env step is stored and followed by the updates it owes.

    One iteration is `cfg.algo.train_every` env steps; it is the unit for logging,
    checkpointing and `env.set_progress`, as in the on-policy loop.
    """
    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_iters = total_frames // frames_per_batch
    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    # also report how episodes that were not terminated ended: time limit or command (motion) finished
    end_keys = [("stats", "episode_time_limit"), ("stats", "command_finished")]
    episode_stats = EpisodeStats(
        list(dict.fromkeys(list(episode_stats.in_keys) + end_keys)), device=env.device
    )
    # Joint training keeps the familiar train/* namespace for student episodes.
    # Use the same components for teacher/*, with a separate episode denominator.
    train_mask = "student_env_mask" if getattr(policy, "student_env_mask", None) is not None else "stats_env_mask"
    stats_groups = {"train": (episode_stats, train_mask)}
    if getattr(policy, "teacher_env_mask", None) is not None:
        stats_groups["teacher"] = (
            EpisodeStats(episode_stats.in_keys, device=env.device), "teacher_env_mask"
        )

    # Most off-policy agents run exactly the invocation's frame budget. A staged
    # agent can finish its current phase and a final SAC tail after that budget.
    configure_budget = getattr(policy, "configure_training_budget", None)
    if configure_budget is not None:
        configure_budget(total_iters)

    def iteration_indices():
        i = 0
        while not (policy.training_complete() if configure_budget is not None else i >= total_iters):
            yield i
            i += 1

    carry = env.reset()
    rollout_policy = policy.get_rollout_policy("train")
    progress = tqdm(iteration_indices(), total=total_iters) if aa.is_main_process() else iteration_indices()
    env_frames = 0
    start_iter = env.current_iter
    for i in progress:
        env.set_progress(start_iter + i)
        rollout_time = training_time = 0.
        stats_buf = []
        for _ in range(cfg.algo.train_every):
            rollout_start = time.perf_counter()
            with torch.no_grad():
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                policy.add_transition(td)
                next_td = td["next"]
                # the env truncates on its time limit or when the command finishes
                time_limit = next_td["step_count"] >= env.max_episode_length
                next_td["stats", "episode_time_limit"] = time_limit.float()
                next_td["stats", "command_finished"] = (next_td["truncated"] & ~time_limit).float()
                # The environment may reuse its statistics buffers at the next step.
                step_stats = td.select(("next", "done"), ("next", "stats")).clone()
                stats_buf.append(step_stats)
            training_start = time.perf_counter()
            policy.update()
            rollout_time += training_start - rollout_start
            training_time += time.perf_counter() - training_start

        rollout_stats = torch.stack(stats_buf, dim=1)
        for accumulator, mask_name in stats_groups.values():
            stats_mask = getattr(policy, mask_name, None)
            selected_stats = rollout_stats
            if stats_mask is not None:
                selected_stats = rollout_stats.clone(False)
                selected_stats["next", "done"] = (
                    rollout_stats["next", "done"] & stats_mask.reshape(env.num_envs, 1, 1)
                )
            accumulator.add(selected_stats)
        env_frames += frames_per_batch

        # A staged policy may replace its recurrent perception at an iteration boundary.
        # Start new episodes so hidden states collected with the old model never enter replay.
        if getattr(policy, "needs_env_reset", False):
            carry = env.reset()
            policy.needs_env_reset = False

        info = {}
        if i % log_interval == 0:
            for prefix, (accumulator, _) in stats_groups.items():
                if len(accumulator):
                    for k, v in sorted(accumulator.pop().items(True, True)):
                        suffix = "/".join(k) if isinstance(k, tuple) else k
                        info[f"{prefix}/{suffix}"] = torch.mean(v.float()).item()
        info.update(policy.pop_info())
        info.update(env.extra)
        info.update(env.stats_ema)

        info["env_frames"] = env_frames
        info["rollout_fps"] = frames_per_batch / rollout_time
        info["training_time"] = training_time

        if should_save(i):
            save(policy, f"checkpoint_{i}")

        if aa.is_main_process():
            run.log(info)



@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    if cfg.algo.get("runner") == "gpo":
        from active_adaptation.learning.gpo.runner import train
        return train(cfg)
    if cfg.algo.get("runner") == "cts":
        from active_adaptation.learning.cts.runner import train
        return train(cfg)
    if cfg.algo.get("runner") == "sitt":
        from active_adaptation.learning.sitt.runner import train
        return train(cfg)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.algo.get("num_envs") is not None:
        cfg.task.num_envs = cfg.algo.num_envs
    # Depth uses CUDA ray casting and does not require the RTX renderer.
    # Keep rendering for explicitly requested evaluation video.
    if "enable_rtx" in cfg.algo:
        cfg.app.enable_cameras = bool(cfg.eval_render or cfg.algo.enable_rtx)
    elif cfg.eval_render:
        cfg.app.enable_cameras = True

    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    env, policy, vecnorm = make_env_policy(cfg)

    import inspect
    import shutil
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if vecnorm is not None and cfg.vecnorm is not None:
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    if getattr(policy, "is_off_policy", False):
        train_off_policy(cfg, env, policy, run, episode_stats, save, should_save)
    else:
        carry = env.reset()
        rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")

        with torch.inference_mode():
            tmp_carry = rollout_policy(carry.clone(False))
            tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
            tmp_td["next"] = tmp_td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)

        N = env.num_envs
        T = cfg.algo.train_every
        device = env.device

        data_buf = TensorDict({}, batch_size=[N, T], device=device)
        for key, value in tmp_td.items(include_nested=True, leaves_only=True):
            shape_tail = value.shape[1:]
            buf = torch.empty((N, T, *shape_tail), dtype=value.dtype, device=device)
            data_buf.set(key, buf)
        logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

        if aa.is_main_process():
            progress = tqdm(range(total_iters))
        else:
            progress = range(total_iters)

        env_frames = 0
        start_iter = env.current_iter
        for i in progress:
            rollout_start = time.perf_counter()
            with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
                torch.compiler.cudagraph_mark_step_begin() # for compiled policy
                env.set_progress(start_iter + i)
                for step in range(cfg.algo.train_every):
                    carry = rollout_policy(carry)
                    td, carry = env.step_and_maybe_reset(carry)
                    td["next"] = td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)
                    data_buf[:, step] = td
                policy.critic(data_buf)
                values = data_buf["state_value"]
                data_buf["next", "state_value"] = torch.where(
                    data_buf["next", "done"],
                    values, # a walkaround to avoid storing the next states
                    torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
                )
            rollout_time = time.perf_counter() - rollout_start

            episode_stats.add(data_buf)
            env_frames += data_buf.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            training_start = time.perf_counter()
            info.update(policy.train_op(data_buf))
            training_time = time.perf_counter() - training_start
            info.update(env.extra)
            info.update(env.stats_ema)

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames
            info["rollout_fps"] = data_buf.numel() / rollout_time
            info["training_time"] = training_time

            if should_save(i):
                save(policy, f"checkpoint_{i}")

            if aa.is_main_process():
                # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
                run.log(info)

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats, policy_trajs = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    run.log(info)

    wandb.finish()
    os._exit(0)
    env.close()
    simulation_app.close()

    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"
    
    return run_path



if __name__ == "__main__":
    main()
