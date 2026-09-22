# GPO with CTS / VAIC perception

```bash
conda activate vaic
python scripts/train.py \
  algo=gpo_ppo_vel_train \
  task=G1/gpo/skateboard_general_tracking
```

The default is **GPO-clip** from *Guided Policy Optimization under Partial
Observability*, equations (9), (10), (12), and Algorithm 1. All 1024 environments
execute the guider. The learner is trained on those same trajectories and is the
default evaluation/deployment policy. Training starts from scratch; no pretrained
teacher or CTS checkpoint is required.

## Architecture

The perception modules, 256-dimensional normalized latents, and shared actor are
the same as `cts_ppo_vel_train`. This is a GPO adaptation to this architecture,
not a reproduction of the paper's Brax MLP or its unified input with role bits.

- Guider: `object_ + object_geo_ -> object_transform`; then
  `[priv, object_, object_trans] -> encoder_priv -> priv_feature -> L2 normalize`.
  The skateboard task has **no height observation and no height CNN**.
- Learner: `depth -> CNN + TemporalDepthGRU -> depth feature`; combine that with
  the `policy + vel_command` MLP in `object_adapt`; transform the predicted object
  using `object_geo_`; then `[policy, vel_command, object_pred, object_pred_trans]`
  passes through `adapt_module` GRU to `priv_pred -> L2 normalize`.
  Depth enters through object prediction, exactly as in CTS/VAIC. The learner
  never reads `priv`, GT `object_`, `command`, or `ref_joint_pos_`.
- Shared actor: `[vel_command, policy, latent]`, MLP `[512, 256, 256]`, existing
  `IndependentNormal` and shared action std. No reference joint-position residual.
- Critic: CTS's privileged critic inputs plus the **guider** latent, detached so
  value regression does not train either perception encoder. Reward-group value
  heads, environment rewards, and VAIC GAE/VecNorm behavior are retained.

`ppo_vel_finetune` uses EMA perception for actions and unnormalized latents;
GPO uses the CTS online modules and normalized latents for both learning and
deployment. There is no latent reconstruction, object MSE, or extra perception
optimizer: learner perception receives gradients from GPO imitation and auxiliary
RL. The retained `object_pred` intermediate is not supervised to reproduce the
physical object state, so its name does not imply a calibrated state estimate.

## Update

Write `mu` for the current guider, `pi` for the current learner, `beta=mu_old`
for the stored behavior distribution, and `sg` for stop-gradient. Both policy
ratios are relative to the **old guider**, including the learner's `pi / beta`.

For GPO-clip, the loss minimized is:

```text
rho_g = mu(a|s) / beta(a|s)
rho_l = pi(a|history) / beta(a|s)
rho_g_clip = clip(clip(mu / sg(pi), 1-delta, 1+delta)
                  * sg(pi) / beta, 1-epsilon, 1+epsilon)
m = indicator(mu/pi outside [1-delta, 1+delta])

L_guider  = -mean(min(rho_g*A, rho_g_clip*A))
            + mean(m * KL(mu || sg(pi)))
L_learner = mean(KL(sg(mu) || pi))
            - alpha * mean(min(rho_l*A, clip(rho_l, 1-epsilon, 1+epsilon)*A))
L_total   = L_guider + L_learner + value_MSE - entropy_coef * H(mu)
```

KL alignment uses a mean over action dimensions, matching the reference
repository; likelihood ratios use summed joint log-probabilities. Double-clip is
computed in log space. The reference learner log-ratio cap of 10 is retained; an
additional guider cap of 60 prevents overflow for pathological inputs. Old and
current guider KL (for optional LR adaptation) is a separate joint-action KL.
Entropy is computed from the **current guider**, allowing an entropy gradient;
the reference's entropy computed from stored behavior logits has no gradient.

One joint policy backward/Adam step updates the shared actor and both encoders;
stop-gradient is applied to distribution outputs, including scales. A separate
critic optimizer has no shared parameters. There is no sequential reconstruction
phase. The learner GRUs advance at every guider step, using the actual executed
action history from the environment, and reset at episode boundaries.

PPO minibatches preserve complete rollout sequences (`[env, time]`) and recompute
learner perception with gradients in every epoch. Initial hidden states are
detached rollout states; the existing adaptation GRU's first-quarter hidden-state
detach behavior is retained. Timeout values bootstrap from the actual terminal
observation, not the auto-reset observation. Guider next-value calculation does
not advance learner history a second time. Initial steps use the existing CTS
`step_count > 1` loss mask.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `algo.gpo.variant` | `clip` | `clip` or `penalty` |
| `algo.gpo.inner_clip` | `0.1` | Guider/learner inner clipping delta |
| `algo.gpo.learner_rl_coef` | `2.0` | Fixed auxiliary RL weight for clip |
| `algo.clip_param` | `0.2` | Outer PPO epsilon, retained from CTS |
| `algo.gpo.target_kl` | `0.001` | Alignment KL threshold for penalty |
| `algo.gpo.penalty_init` | `0.0` | Initial adaptive alpha for penalty |
| `algo.gpo.penalty_factor` | `1.2` | Epoch-wise multiplier/divisor |
| `algo.gpo.penalty_max` | `10.0` | Adaptive alpha cap |
| `algo.desired_kl` | `null` | Fixed LR by default, as in reference GPO |
| `algo.gpo.normalize_latent` | `true` | CTS latent normalization |
| `algo.gpo.evaluate_at_end` | `true` | Evaluate learner after final checkpoint |

`train_every=32`, `ppo_epochs=3`, `num_minibatches=8`, `lr=3e-4`, `gamma=0.99`,
and `lmbda=0.95` retain the CTS/VAIC baseline settings. These settings have not been
tuned for skateboard GPO. `num_envs` must be divisible by `num_minibatches` during
training. Single-process/GPU training is supported.

GPO-penalty uses the ordinary guider PPO clip and adaptive `alpha` on **both** the
guider backtracking KL and learner auxiliary PPO loss (paper equation 11).
The imitation KL coefficient remains one. Adaptation occurs after each PPO epoch,
including the reference startup from zero (`alpha=0.1` when above target).

```bash
python scripts/train.py algo=gpo_ppo_vel_train \
  task=G1/gpo/skateboard_general_tracking algo.gpo.variant=penalty
```

## Logging and checkpoints

W&B uses the existing `ppo_vel_train` dashboard keys, with no additional `gpo/*`
metrics. The command uses the same `cfg.wandb` project/mode settings as PPO.

- `train/stats/*`: completed training episodes, all driven by guider.
- `reward.<group>/<term>`: training reward EMAs, unchanged environment keys.
- `actor/policy_loss`: guider's double-clipped RL surrogate (ordinary PPO
  surrogate for the penalty variant), not the sum of all GPO losses.
- `actor/kl`, `actor/approx_kl`, `actor/clamp_ratio`: old/current guider policy
  diagnostics. Guider/learner alignment KL remains internal to GPO.
- Other `actor/*`, `critic/*`, `actor_std/*`: PPO's existing gradient norms,
  entropy/std, value loss, explained variance, return and reward diagnostics.
- `env_frames`, `rollout_fps`, `training_time`, `performance*`: existing timing keys.
- `eval/*`: deterministic learner-only evaluation, one first completed episode
  per environment, normalizer frozen. `episode_cnt` reports completed episodes;
  incomplete episodes are reported only in the terminal.

GPO-specific losses and the adaptive penalty are used for training but are not
published as additional metrics. Reconstruction `adapt/*` losses are absent
because GPO does not optimize those objectives. Existing W&B runs retain their
historical keys; the schema applies to subsequent logs.

The learner's improvement cannot be inferred from guider training return alone.
Use the saved checkpoints for learner evaluation, or the final `eval/*` metrics.

Resume with the same task/algorithm settings and an increased **total** frame
budget if needed:

```bash
python scripts/train.py algo=gpo_ppo_vel_train \
  task=G1/gpo/skateboard_general_tracking \
  checkpoint_path=/absolute/path/checkpoint_300.pt
```

`run:<entity>/<project>/<run_id>` is also accepted. Checkpoints save models, both
optimizers, VecNorm, iteration, learning rate, adaptive penalty, config and RNG.
Simulator state is not resumed; collection starts from new episodes. CTS/SITT/VAIC
checkpoints are rejected. `algo.gpo.eval_policy` and `evaluate_at_end` may change
on resume; task and other algorithm settings must agree.

`policy.get_rollout_policy("deploy")` returns a self-contained learner module
without guider/critic or an owner reference. Apply the saved observation
normalizer to its inputs. `get_rollout_policy("eval")` defaults to learner;
`algo.gpo.eval_policy=guider` enables a comparison. Final training evaluation always
uses learner regardless of this setting.

## Verification

```bash
python -m unittest discover -s tests/gpo -v
GPO_TEST_DEVICE=cuda python -m unittest discover -s tests/gpo -v

python scripts/train.py algo=gpo_ppo_vel_train \
  task=G1/gpo/skateboard_general_tracking \
  task.num_envs=8 task.max_episode_length=16 \
  algo.train_every=16 algo.num_minibatches=2 algo.ppo_epochs=1 \
  total_frames=256 save_interval=1 wandb.mode=disabled
```

Tests cover analytical clip/penalty objectives, signed advantages, double-clip
gradients, both KL detach directions, action-dimension reductions, complete
perception gradients, CTS network parity, privilege isolation, sequential GRU
replay/reset, terminal bootstrap, joint update, optimizer continuation, standalone
deployment, task configuration parity, and evaluation statistics.

References: [paper](https://arxiv.org/abs/2505.15418), local author implementation
`/home/hcc/research/GPO/GPO.py` at commit `d7a955ddff9ccb7e447f74e78a177791a1abcc9d`.
