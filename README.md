# VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands

<div align="center">
<a href="https://vaic-humanoid.github.io/">
	<img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://arxiv.org/abs/2606.09286">
	<img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/ldt29/VAIC/stargazers">
	<img alt="GitHub stars" src="https://img.shields.io/github/stars/ldt29/VAIC?style=social"/>
</a>

</div>

This repository hosts the open-source release for the paper VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands.


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n vaic python=3.11 -y
conda activate vaic

# install isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# install isaaclab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none

# install vaic
cd ..
git clone https://github.com/ldt29/VAIC
cd VAIC
pip install -e .
```

## Verify Your Data
Visualize motions in Isaac Sim with `task.command.replay_motion=true`:

```bash
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea task.command.replay_motion=true
```


## Train and Evaluate

Teacher policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_train task=G1/vaic/skateboard_tea
# evaluate policy
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<wandb-run-path>
```

Student policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
```
To export trained policies, add `export_policy=true` to the play script.

GPO training with the CTS guider/learner architecture:

```bash
python scripts/train.py \
  algo=gpo_ppo_vel_train \
  task=G1/gpo/skateboard_general_tracking
```

The default is GPO-clip: all environments execute the guider, and the learner's
depth/object/GRU perception is trained jointly through the GPO objective.
See [GPO settings, checkpoint resume and verification](active_adaptation/learning/gpo/README.md).

One-stage FlashSAC (Linux ARM64 DGX Spark and x86_64 NVIDIA desktops):

```bash
python scripts/train.py \
  algo=one_stage_flashsac_vel_train \
  task=G1/flashsac/skateboard_general_tracking
```

Teacher latent pull loss is enabled by default with a maximum weight of `0.001`,
perception warm-up, a linear ramp, and a gradient norm cap. Disable it with
`algo.teacher_pull_enabled=false`. See the
[Korean explanation and settings](active_adaptation/learning/flashsac/README.md#teacher-latent-pull-loss-기본-활성화).

Use Isaac Sim 5.1 and Isaac Lab **v2.3.2** as in the setup above. Isaac Lab
v2.3.0 lacks the multi-mesh ray caster required by the depth sensor. On Spark,
keep the CUDA-enabled ARM64 PyTorch build supplied with your environment;
installing this repository with `pip install --no-deps -e .` preserves it
when the other dependencies are already installed.

The training entry point automatically preloads OpenMP on Linux ARM64 and
initializes Isaac Sim before TorchRL selects its multiprocessing method.
It preserves the training worker start method; x86_64 skips this bootstrap.
Hydra `--cfg job` and `--help` do not initialize the simulator.

For a small end-to-end check with logging disabled:

```bash
python scripts/train.py \
  algo=one_stage_flashsac_vel_train \
  task=G1/flashsac/skateboard_general_tracking \
  wandb.mode=disabled algo.num_envs=4 \
  algo.buffer_max_length=512 algo.buffer_min_length=256 \
  algo.sample_batch_size=64 algo.updates_per_interaction_step=1 \
  total_frames=512 task.max_episode_length=64 save_interval=-1
```

The full defaults use 4096 environments and a 10-million-transition CPU replay
buffer. Adjust `algo.num_envs` and `algo.buffer_max_length` for available RAM;
the replay must hold at least 81 steps per environment with the default history
settings. See [FlashSAC settings](active_adaptation/learning/flashsac/README.md).


## Acknowledgments

This repository is built on top of [HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos](https://github.com/LeCAR-Lab/HDMI). We thank the authors for open-sourcing their work.

## Citation

If you find our work useful for your research, please consider citing us:

```bibtex
@article{li2026vaic,
  title = {VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands},
  author = {Li, Dongting and Wu, Qianyang and Chen, Xingyu and Li, Liang and Lin, Yuhang and Wu, Sikai and Zhang, Guoyao and Zhou, Mingliang and Xiang, Diyun and Zhang, Qiang and Xu, Renjing and Ma, Jianzhu},
  journal = {arXiv preprint arXiv:2606.09286},
  year = {2026}
}
```
