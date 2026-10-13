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
  Teacher encoder는 teacher SAC loss와 아래의 작은 latent pull loss로 학습합니다.
- Student perception은 **작은 CNN + 고정 길이 causal window Transformer**를
  기본으로 사용합니다. GRU, recurrent hidden, hidden checkpoint와 burn-in은
  사용하지 않습니다. 기본 구조는 L=16, D=128, 1층, 4 heads, FFN=256입니다.

```text
depth → Conv(1→8, kernel=5, stride=4) → Mish
      → Conv(8→8, kernel=3, stride=2) → Mish → Linear → feature(64)
      → Linear → visual(256)
policy observation + velocity command + object-frame geometry
                                      → MLP → context(256)
[context, visual] → Linear → token(128)
최근 16개 token + window 내 위치 embedding → causal Transformer 1층
현재 시점 출력 → LayerNorm ┬→ object head
                         └→ latent head(256)
```

Object geometry는 object 자체 좌표계의 고정 template입니다. Student는
실제 object pose로 변환한 geometry를 입력받지 않습니다. Reset 또는 invalid
placeholder 이전의 관측은 attention에서 제외합니다. 초기 history는 padding을
mask하고, invalid query의 출력은 0으로 처리합니다.

한 프레임의 CNN/context 특징은 chunk 안에서 한 번 계산하고 겹치는 window가
공유합니다. 각 예측은 **정확히 자기 L개 프레임만** 사용하므로 층을 늘려도
history가 다른 window로 확장되지 않습니다. 마지막 attention 층은 현재 시점의
query만 계산하며, 전체 causal 층의 마지막 출력을 선택하는 것과 동일합니다.

Rollout에서는 raw depth/context의 ring buffer와 CNN token을 유지합니다.
Perception optimizer가 가중치를 바꾸면 다음 rollout에서 raw history 전체를
현재 가중치로 다시 encode합니다. 이 캐시는 학습된 recurrent state가 아닙니다.
Teacher 환경에서는 perception을 실행하지 않고 student 절반에서만 실행합니다.
Teacher depth도 replay에 저장하여 student perception의 학습 데이터로 사용합니다.

Perception은 `priv_loss_coef × MSE(latent, teacher_latent.detach()) +
object_loss_coef × MSE(object_prediction, object_target)`로만 학습합니다.
Teacher latent는 replay를 sampling한 시점의 teacher encoder로 다시 계산합니다.
CNN, Transformer와 두 head는 하나의 perception optimizer를 공유합니다.
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

## Teacher latent pull loss (기본 활성화)

현재 `one_stage_flashsac_vel` 및 `one_stage_flashsac_vel_train`은 teacher의
`encoder_priv`가 student가 추정하기 쉬운 latent를 만들도록 약하게 정규화합니다.
기존 실행 명령에 추가 옵션을 넣지 않아도 활성화됩니다.

```text
teacher loss = teacher SAC loss + effective_coef × mean((z_teacher - z_student.detach())²)
student perception loss = priv_loss_coef × mean((z_student - z_teacher.detach())²)
                        + object_loss_coef × object_loss
```

Pull target은 같은 replay transition의 현재 student perception 출력입니다.
현재 가중치의 관측 window, episode/valid mask와 teacher/student 50:50 sampling을
사용합니다. Pull loss는 `encoder_priv`에만 gradient를 보내며, student
perception, teacher의 policy head, critic, 선택적인 height encoder에는 직접
gradient를 보내지 않습니다. Teacher SAC loss와 **동일한 optimizer step**에서
합쳐 처리하고, actor update를 건너뛰는 step에는 pull도 적용하지 않습니다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `teacher_pull_enabled` | `true` | Teacher pull 활성화 |
| `teacher_pull_coef` | `0.001` | 선형 증가 후의 최대 MSE 가중치 |
| `teacher_pull_warmup_updates` | `100` | 초기 perception update 동안 가중치 0 |
| `teacher_pull_ramp_updates` | `1000` | 이후 최대 가중치까지 증가할 perception update 수 |
| `teacher_pull_max_grad_ratio` | `0.1` | Encoder의 RL gradient norm 대비 pull gradient norm 상한 |

Schedule은 수집 frame이나 SAC update가 아니라 **perception optimizer update
횟수** 기준입니다. 기본 설정에서 100회까지 가중치가 0이고, 600회에서는
0.0005, 1100회부터는 0.001입니다. Resume 시 저장된 perception update
counter를 이어 사용합니다. 새 학습은 0부터 시작합니다.
`teacher_pull_ramp_updates=0`이면 warm-up 종료 후 바로 최대 가중치를 씁니다.

작은 가중치만으로 gradient가 작다는 보장은 없으므로 매 teacher actor update에서
encoder의 RL gradient norm과 pull gradient norm을 비교해 실제 가중치를
더 낮출 수 있습니다. RL gradient가 0이면 pull도 0입니다. 이 10% 제한은
**Adam 적용 전 raw gradient norm** 기준이며, parameter 변화나 성능 저하를
10% 이내로 보장하는 것은 아닙니다. AMP에서는 scaled gradient를 측정한 뒤
unscale하여 비교합니다. 이 측정에는 추가 backward 계산 비용이 있습니다.

W&B의 `sac/teacher/actor/` 아래에서 다음을 확인할 수 있습니다.

- `pull_scheduled_coef`: 현재 warm-up/ramp 가중치.
- `pull_coef`, `pull_effective_coef`: actor update 당시 예정/실제 가중치.
- `pull_loss`, `pull_weighted_loss`: 원래 MSE와 가중치를 적용한 값.
- `sac_loss`, `loss`: 기존 actor objective와 pull을 더한 전체 objective.
- `pull_grad_norm`, `rl_encoder_grad_norm`, `pull_grad_ratio`: gradient 크기와 적용 비율.
- `teacher_latent_std`, `student_latent_std`: batch 내 latent 변화량의 차원별 표준편차 평균.

Pull 관련 update 지표는 가중치가 양수인 teacher actor update에서 기록됩니다.
`pull_scheduled_coef`는 warm-up 중에도 기록됩니다. Latent MSE가 줄어도 두
latent가 상수에 가까워진 것일 수 있으므로 teacher/student tracking 성능과
latent 표준편차를 함께 비교해야 합니다. 기본 수치는 보수적인 실험 시작값이며,
장기 학습 성능 개선은 검증되지 않았습니다.

기존 방식과 비교하려면 같은 명령에 다음을 추가합니다.

```bash
algo.teacher_pull_enabled=false
```

`algo.teacher_pull_coef=0`도 pull을 완전히 끕니다. 테스트용 짧은 run에서
pull 경로를 확인하려면 `algo.teacher_pull_warmup_updates=0
algo.teacher_pull_ramp_updates=2`를 사용할 수 있습니다.

## Window 설정과 replay

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `perception_window_size` | 16 | 현재 프레임을 포함하는 관측 window 길이 |
| `perception_model_dim` | 128 | Transformer token 차원 |
| `perception_num_layers` | 1 | Causal Transformer 층 수 |
| `perception_num_heads` | 4 | Attention head 수; token 차원을 나누어야 함 |
| `perception_ffn_dim` | 256 | 각 Transformer 층의 FFN 중간 차원 |
| `latent_dim` | 256 | Student actor에 전달하는 latent 차원 |
| `sequence_length` | 32 | Replay chunk마다 loss를 계산하는 transition 수 |
| `n_step` | 3 | SAC reward/bootstrap horizon |
| `depth_num_bins` | 100 | Depth sensor의 기존 양자화와 일치 |
| `perception_updates_per_iteration` | 1 | 학습 시작 후 iteration당 perception update 수 |
| `train_every` | 32 | Iteration당 환경 step 수 |
| `updates_per_interaction_step` | 8 | 환경 step당 공동 SAC update 수 |
| `sample_batch_size` | 2048 | SAC transition 수; `2 × sequence_length`의 배수 |
| `buffer_max_length` | 10,000,000 | 요청 transition 용량; 전체 환경 row 단위로 내림 |
| `buffer_min_length` | 100,000 | Update 시작 전 최소 수집량 |
| `buffer_obs_dtype` | `float16` | Vector observation 저장 dtype |
| `buffer_device_type` | `cpu` | Replay는 host RAM에 저장 |

한 chunk는 **이전 관측 15 + 학습 관측 32 + bootstrap 관측 3 = 50개 관측**입니다.
Buffer는 최소 `50 × num_envs` transition을 담아야 합니다. `sequence_length`는
window 길이가 아니라 한 chunk에서 학습할 transition 수입니다. L=16이고 제어
주기가 0.02초이면 각 예측은 현재부터 최대 0.30초 전의 관측을 직접 봅니다.

Depth와 vector observation은 매 step 한 번 저장하고, 샘플링 시 필요한 원본
구간을 읽습니다. Window를 transition마다 중복 저장하지 않으므로 replay 용량을
L배로 늘리지 않습니다. Hidden/latent 저장과 burn-in은 완전히 제거했습니다.
Supervised loss는 window의 이전 프레임 CNN까지 역전파하며, SAC용 latent는
현재 perception으로 `no_grad()` 재계산합니다.

학습 시작 후 완전한 iteration은 환경 32 step × SAC 8 update = **공동 SAC
update 256회, perception update 1회**입니다. Actor는 공동 update 두 번마다
갱신합니다. 최소 replay 수집 전에도 teacher/student 분할로 rollout합니다.
기본 `algo.num_envs=4096`이 task의 환경 수를 덮어씁니다.

Episode 종료 시 실제 next observation/depth를 별도로 보관합니다. Truncation
bootstrap window에는 종료 직전의 L-1개 관측과 실제 종료 관측을 넣습니다.
다음 episode의 reset 관측은 사용하지 않습니다. Termination은 bootstrap을
차단하고 soft discount와 gamma는 n-step horizon에 한 번씩 반영합니다.

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
P = depth 채널 × 높이 × 너비
b = vector dtype 바이트 수 (float16: 2, float32: 4)

fixed bytes = C × (bD + P + 4A + 12) + 8T + 16(T + 1)
```

`16(T + 1)`은 출처별 valid-count metadata입니다. Hidden 저장 공간은 없습니다.
**Buffer에 보존된 종료 transition마다** 추가 메모리가
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
  task=G1/flashsac/skateboard_general_tracking \
  headless=true eval_render=false wandb.mode=disabled vecnorm=null \
  algo.num_envs=64 \
  algo.buffer_max_length=8192 algo.buffer_min_length=4096 \
  algo.sample_batch_size=256 algo.updates_per_interaction_step=1 \
  algo.use_compile=false algo.use_amp=false \
  total_frames=16384 save_interval=-1
```

위 replay는 환경당 128 step을 담으므로 최소 50 step 조건을 만족합니다.
`algo.num_envs`가 task의 환경 수를 덮어씁니다. `vecnorm=null`로 기존 raw
observation을 사용합니다. `enable_rtx=false`가 기본이며 ray-cast depth는
계속 활성화됩니다. `eval_render=true`이면 rendering이 다시 켜집니다.
Training runner는 마지막에 evaluation과 최종 checkpoint 저장도 수행합니다.
짧은 실행은 오류 확인용이며, 장기 추적 성능은 별도 검증이 필요합니다.

기본 대규모 설정을 사용하려면 smoke-run용 `algo.*`와 `total_frames`
override를 제거하되, 먼저 실제 observation 크기에 따른 RAM 요구량을
확인해야 합니다. 기본 replay는 1천만 transition 전체 용량을 할당합니다.

## Checkpoint와 관찰 지표

- `checkpoint_path=null`: teacher, student actor, window perception을 새로 학습합니다.
- 이 구조로 저장한 checkpoint만 resume할 수 있습니다. 두 actor, Q/target,
  temperature, perception, optimizer/scheduler 및 counter를 복원합니다.
- 이전 GRU one-stage checkpoint와 teacher-only checkpoint는 지원하지 않습니다.
  기존 checkpoint를 변환하거나 부분적으로 불러오는 경로는 없습니다.

Resume 시 replay와 진행 중이던 episode, 탐색 noise, raw window 및 token cache는
새로 시작합니다. 모델·window 설정, observation layout, geometry, action convention은
일치해야 합니다. Resume는 bitwise 동일한 학습 연속 실행이 아닙니다.

Runner는 student rollout 통계만 `train/stats/*`에, 동일한 항목의 teacher
rollout 통계만 `teacher/stats/*`에 기록합니다. 각 출처에서 종료된 episode 수로
독립적으로 평균하며, 종료된 episode가 없는 출처는 해당 시점에 기록하지 않습니다.
Actor/temperature 학습 지표는 `sac/teacher/*`, `sac/student/*`로 구분합니다.
기본 evaluation은 student를 사용합니다. 알고리즘의
`get_rollout_policy("teacher")`는 teacher 평가용 policy를 제공합니다.
`perception/priv_loss`, `perception/object_loss`, `perception/grad_norm`과
actor별 entropy/temperature를 함께 확인할 수 있습니다.

CPU 테스트는 simulator를 시작하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' TORCHDYNAMO_DISABLE=1 python -m pytest tests/flashsac -q
```

Vendored FlashSAC의 출처와 라이선스는 `flashsac_upstream/SOURCE.json` 및
`flashsac_upstream/LICENSE`에 있습니다.
