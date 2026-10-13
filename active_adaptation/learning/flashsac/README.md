# One-stage FlashSAC velocity training

`one_stage_flashsac_vel.py`는 teacher와 student를 같은 학습 루프에서
학습합니다. Hydra 이름은 `one_stage_flashsac_vel`이며
`one_stage_flashsac_vel_train`도 같은 설정의 alias입니다.
구현 클래스는
`active_adaptation.learning.flashsac.one_stage_flashsac_vel.OneStageFlashSACVel`,
설정 클래스는 같은 모듈의 `OneStageFlashSACVelConfig`입니다.

## 모델과 학습 신호

- Teacher는 기존 `FlashSACVel`의 `encoder_priv`, privileged input 구성,
  object geometry 변환, 선택적인 height encoder를 유지합니다.
  Teacher encoder는 teacher SAC loss로 계속 학습합니다.
- Student perception은 scratch부터 시작합니다. 기존 두 GRU 대신 다음
  gate + shared belief GRU를 사용합니다.

```text
depth → CNN × 3 → feature(64) → Linear → visual(256)
policy observation + velocity command + object-frame geometry
                                      → MLP → context(256)
[context, visual, previous belief] → Linear → sigmoid → gate(256)
[context, gate × visual] → GRUCell(512 → 256) → belief
                                              ├→ object head
                                              └→ latent head(256)
```

Object geometry는 object 자체 좌표계의 고정 template입니다. Student는
실제 object pose로 변환한 geometry를 입력받지 않습니다. Episode reset은
gate 계산 전에 hidden을 초기화합니다.

Perception은 `priv_loss_coef × MSE(latent, teacher_latent.detach()) +
object_loss_coef × MSE(object_prediction, object_target)`로만 학습합니다.
Teacher latent는 replay를 sampling한 시점의 teacher encoder로 다시 계산합니다.
두 head와 shared GRU는 하나의 perception optimizer를 공유합니다.
Student actor는 `velocity command + policy observation + latent.detach()`를
입력받습니다. **Student SAC loss는 perception으로 역전파하지 않습니다.**

Teacher/student actor는 각각 SAC로 학습하고, Q1/Q2 및 target Q1/Q2는 한
세트만 공유합니다. Entropy temperature는 actor별로 둡니다. Shared Q의
bootstrap target은 각 actor의 action/entropy로 만든 categorical target을
각각 projection한 뒤 50:50으로 평균합니다. 따라서 두 정책을 혼합한
continuation을 학습합니다. Q optimizer와 target EMA는 공동 update당
한 번씩 갱신됩니다.

환경의 앞 절반은 teacher, 뒤 절반은 student가 제어하므로 `num_envs`는
짝수여야 합니다. Replay 용량도 각 출처에 절반씩 배정됩니다. 두 actor는
같은 혼합 batch로 학습합니다. SAC batch는 유효 transition을 출처별로
절반씩 뽑고, perception loss는 유효 transition 수가 달라도 각 출처의
총 가중치가 0.5가 되도록 계산합니다.

## Replay와 recurrent history

기본 설정은 다음과 같습니다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `checkpoint_interval` | 16 | 각 환경의 16 step마다 observation 처리 **전** hidden 저장 |
| `burn_in` | 16 | 최소 burn-in; checkpoint 정렬에 따라 실제 16–31 step |
| `sequence_length` | 32 | Chunk마다 loss를 계산할 학습 step 수 |
| `n_step` | 3 | SAC n-step reward/bootstrap horizon |
| `belief_hidden_dim` | 256 | 하나의 recurrent hidden 크기 |
| `depth_num_bins` | 100 | 기존 depth sensor 양자화와 일치해야 함 |
| `perception_updates_per_iteration` | 1 | 학습 시작 후 iteration당 perception update 수 |
| `train_every` | 32 | Training iteration당 환경 step 수 |
| `updates_per_interaction_step` | 8 | 학습이 시작된 뒤 환경 step당 공동 SAC update 수 |
| `sample_batch_size` | 2048 | SAC transition 수; `2 × sequence_length`의 배수 |
| `buffer_max_length` | 10,000,000 | 요청 transition 용량; 전체 환경 row 단위로 내림 |
| `buffer_min_length` | 100,000 | Update 시작 전 최소 수집량 |
| `buffer_obs_dtype` | `float16` | Vector observation 저장 dtype |
| `buffer_device_type` | `cpu` | Replay는 host RAM에 저장 |

한 chunk에는 최대 burn-in 31 + 학습 32 + bootstrap 3 = **66개 관측**을
읽습니다. Ring wrap 이후에도 checkpoint를 보존하려면 buffer가 최소
**81 step × num_envs transition**을 담아야 합니다. 유효 chunk가 충분하고
`buffer_min_length`도 충족해야 학습이 시작됩니다. **첫 rollout부터** 고정된
teacher/student 환경 분할을 유지하고 각 actor의 exploration action을
사용합니다. Warm-up은 update를 시작하기 위한 수집 기간이며, 그동안에도
두 actor로 rollout하고 perception hidden을 기록합니다.

Depth와 vector observation은 매 step 한 번 저장합니다. Hidden은
float16 checkpoint로만 저장합니다. Sampling 시 현재 perception으로
burn-in을 계산하고, 학습 구간에서만 supervised gradient를 전개합니다.
SAC용 latent는 전체 구간을 `no_grad()`로 다시 계산합니다. Perception은
같은 iteration에서 갱신되며 그 사이의 SAC update 동안 가중치를 유지합니다.

기본 설정에서 학습 시작 후 완전한 한 iteration은 환경 32 step × SAC
8 update = **공동 SAC update 256회, perception update 1회**입니다.
Actor는 `actor_update_period=2`에 따라 공동 update 두 번마다 갱신합니다.
`updates_per_interaction_step`과 `perception_updates_per_iteration`으로
상대 학습 빈도를 조절할 수 있습니다. 이 기본 비율의 학습 수렴과 성능은
아직 검증하지 않았습니다.

**저장 hidden + 짧은 burn-in은 현재 모델로 episode 전체를 다시 계산한
결과에 대한 근사입니다.** Perception이 바뀐 후의 online hidden에도 이
불일치가 남을 수 있습니다. Buffer 전체를 16 iteration마다 재계산하거나
FIFO 교체만 기다리는 방식은 사용하지 않습니다. Checkpoint version/age,
latent/object loss, gate 분포를 기록하여 이 영향을 관찰할 수 있습니다.

Episode 종료 시의 실제 next observation/depth는 별도로 보관합니다.
Truncation bootstrap은 reset observation이 아니라 해당 종료 관측과
종료 직전 belief를 사용합니다. Termination 및 bootstrap 설정에 따른
discount는 replay가 n-step horizon에 반영합니다.

## Depth 정밀도와 메모리

현재 depth 전처리는 범위 clipping/정규화 후 `floor(depth * 100) / 100`을
적용하므로 `0.00, 0.01, ..., 1.00`의 **101개 값**이 존재합니다.
Replay는 `round(depth * 100).uint8()`로 bin 번호를 저장하고 학습 시
`stored.float() / 100`으로 복원합니다. 이 입력에 대해서는 uint8 저장이
추가 양자화 손실을 만들지 않습니다. `/255` encoding이 아닙니다.
Sensor의 `num_bins`와 `algo.depth_num_bins`는 반드시 같아야 합니다.
원본 연속 depth를 그대로 보존하는 저장 방식은 아닙니다.

`G1/vaic/skateboard_general_tracking_stu`의 실제 camera 설정은 **64×36**이며,
모델과 replay가 observation spec에서 해상도를 읽습니다. 64×32로 고정하지
않습니다. 1천만 frame의 depth만 계산하면 uint8은 23.04 GB,
float16은 46.08 GB입니다. 64×32에서는 각각 20.48/40.96 GB입니다.
7-bit packing은 구현하지 않았습니다.

실제 고정 tensor allocation은 다음 식으로 계산할 수 있습니다.

```text
E = num_envs, T = floor(buffer_max_length / E), C = T × E
D = 저장 vector observation 차원, A = action 차원
P = depth 채널 × 높이 × 너비, H = hidden 차원
b = vector dtype 바이트 수 (float16: 2, float32: 4)
K = ceil(T / checkpoint_interval) + 1

fixed bytes = C × (bD + P + 4A + 12) + 8T + 16(T + 1) + K × (2EH + 16)
```

`16(T + 1)`은 출처별 valid-count metadata입니다. 마지막 항에는 sparse hidden checkpoint와 그 step/version metadata가
포함됩니다. 1천만 transition, H=256, 간격 16이면 hidden payload는 대략
0.32 GB입니다. **Buffer에 보존된 종료 transition마다** 추가 메모리가
필요합니다. 종료 transition당 대략
`bD + P + 8` bytes이며 Python container overhead도 별도입니다.
Sampled chunk, GPU transfer, backward activation, 모델/optimizer 메모리는
위 식에 포함되지 않습니다. `replay/bytes`는 현재 buffer tensor payload를
기록합니다.

## 실행

저장소 루트에서 VAIC/Isaac Lab 환경을 활성화한 뒤 실행합니다. 아래는
buffer 크기와 update 수를 줄인 smoke-run 설정 예시입니다.

```bash
conda activate vaic
python scripts/train.py \
  algo=one_stage_flashsac_vel \
  task=G1/vaic/skateboard_general_tracking_stu \
  headless=true eval_render=false wandb.mode=disabled vecnorm=null \
  algo.num_envs=64 \
  algo.buffer_max_length=8192 algo.buffer_min_length=4096 \
  algo.sample_batch_size=256 algo.updates_per_interaction_step=1 \
  algo.use_compile=false algo.use_amp=false \
  total_frames=16384 save_interval=-1
```

위 replay는 환경당 128 step을 담으므로 최소 81 step 조건을 만족합니다.
`algo.num_envs`가 task의 환경 수를 덮어씁니다. `vecnorm=null`로 기존 raw
observation을 사용합니다. `enable_rtx=false`가 기본이며 ray-cast depth는
계속 활성화됩니다. `eval_render=true`이면 rendering이 다시 켜집니다.
Training runner는 마지막에 evaluation과 최종 checkpoint 저장도 수행합니다.
이 명령의 GPU/Isaac 실행과 학습 성능은 아직 검증하지 않았습니다.

기본 대규모 설정을 사용하려면 smoke-run용 `algo.*`와 `total_frames`
override를 제거하되, 먼저 실제 observation 크기에 따른 RAM 요구량을
확인해야 합니다. 기본 replay는 1천만 transition 전체 용량을 할당합니다.

## Checkpoint와 관찰 지표

- `checkpoint_path=null`: teacher, student actor, perception 모두 새로 시작합니다.
- `checkpoint_path=/absolute/path/to/teacher.pt`: 호환되는 `flashsac_vel_train`
  checkpoint의 teacher/Q/target 및 teacher temperature 가중치를 불러옵니다.
  **Student actor와 perception은 새로 초기화하고 optimizer도 새로 시작합니다.**
- One-stage checkpoint를 같은 경로 옵션으로 지정하면 두 actor, Q/target,
  temperature, perception, optimizer/scheduler 및 학습 counter를 복원합니다.
  Observation layout, geometry, action convention, depth/history schema가
  일치해야 합니다.

두 load 방식 모두 replay는 비어 있는 상태에서 시작하고 다시 채웁니다.
진행 중이던 episode, rollout noise와 hidden history는 이어서 복원하지
않으므로 checkpoint resume는 bitwise 동일한 학습 연속 실행이 아닙니다.
Teacher warm-start 시 새로운 perception은 기존 GRU checkpoint를 사용하지
않습니다.

Runner는 student rollout 통계만 `train/stats/*`에, 동일한 항목의 teacher
rollout 통계만 `teacher/stats/*`에 기록합니다. 각 출처에서 종료된 episode 수로
독립적으로 평균하며, 종료된 episode가 없는 출처는 해당 시점에 기록하지 않습니다.
Actor/temperature 학습 지표는 `sac/teacher/*`, `sac/student/*`로 구분합니다.
기본 evaluation은 student를 사용합니다. 알고리즘의
`get_rollout_policy("teacher")`는 teacher 평가용 policy를 제공합니다.
`perception/priv_loss`, `perception/object_loss`, `perception/checkpoint_age`,
gate 관련 지표와 actor별 entropy/temperature를 함께 확인할 수 있습니다.

CPU 테스트는 simulator를 시작하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' TORCHDYNAMO_DISABLE=1 python -m pytest tests/flashsac -q
```

Vendored FlashSAC의 출처와 라이선스는 `flashsac_upstream/SOURCE.json` 및
`flashsac_upstream/LICENSE`에 있습니다.
