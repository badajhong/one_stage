"""CTS-only metric splitting; student keys retain the VAIC dashboard schema."""

import torch


class EpisodeMetrics:
    """Same completed-episode reduction as VAIC's EpisodeStats, per group."""

    def __init__(self, env_slice, prefix):
        self.env_slice = env_slice
        self.prefix = prefix
        self.sums = {}
        self.episodes = 0

    def add(self, data):
        nxt = data[self.env_slice]["next"]
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
        info = {f"{self.prefix}/{k}": v / self.episodes for k, v in self.sums.items()}
        self.sums.clear()
        self.episodes = 0
        return info


class RewardMetrics:
    """Observe existing reward computations once, preserving their return values.

    VAIC's global reward EMA has already lost environment IDs. Instance-local
    compute wrappers observe the unaggregated result, including active masks,
    to reproduce that EMA separately. No reward function is recomputed, and no
    reward, mask, weight, enabled flag, or simulator statistic is changed.
    Remove the wrappers before evaluation and on all exit paths.
    """

    def __init__(self, base_env, num_teacher):
        self.decay = base_env._stats_ema_decay
        self.num_envs = base_env.num_envs
        self.slices = {"student": slice(num_teacher, None), "teacher": slice(None, num_teacher)}
        self.ema = {}
        self.originals = []
        try:
            for group, rewards in base_env.reward_groups.items():
                for term, func in rewards.funcs.items():
                    original = func.compute
                    self.originals.append((func, "compute" in vars(func), original))

                    def observe(*args, _original=original, _func=func, _key=(group, term), **kwargs):
                        result = _original(*args, **kwargs)
                        self.record(_key, result, _func.weight)
                        return result

                    func.compute = observe
        except Exception:
            self.close()
            raise

    @torch.no_grad()
    def record(self, key, result, weight):
        if isinstance(result, tuple):
            reward, active = result
            reward = weight * (reward * active.float())
        else:
            reward, active = weight * result, None
        if reward.ndim == 0 or reward.shape[0] != self.num_envs:
            raise ValueError(f"CTS reward logging requires a per-env reward: {key}")
        if active is not None and (active.ndim == 0 or active.shape[0] != self.num_envs):
            raise ValueError(f"CTS reward logging requires a per-env active mask: {key}")
        for role, env_slice in self.slices.items():
            values = reward[env_slice]
            count = values.numel() if active is None else active[env_slice].sum()
            ema_key = (role, *key)
            if ema_key not in self.ema:
                self.ema[ema_key] = (values.new_zeros(()), values.new_zeros(()))
            numerator, denominator = self.ema[ema_key]
            numerator.mul_(self.decay).add_(values.sum())
            denominator.mul_(self.decay).add_(count)

    def snapshot(self):
        info = {}
        for (role, group, term), (numerator, denominator) in self.ema.items():
            # Undefined until this group has at least one active sample.
            if denominator.item() > 0:
                key = (f"reward.{group}/{term}" if role == "student"
                       else f"cts_teacher/reward/{group}/{term}")
                info[key] = (numerator / denominator).item()
        return info

    def close(self):
        for func, had_override, original in reversed(self.originals):
            if had_override:
                func.compute = original
            else:
                del func.compute
        self.originals.clear()
