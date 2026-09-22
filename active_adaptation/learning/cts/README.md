# VAIC perception을 사용하는 CTS

```bash
conda activate vaic
python scripts/train.py \
  algo=cts_ppo_vel_train \
  task=G1/cts/skateboard_general_tracking
```

기본 1024개 환경 중 앞 512개는 teacher, 뒤 512개는 student가 행동한다.
환경 역할은 학습 중 고정한다. 두 그룹은 하나의 actor, action std, critic을 공유한다.
기본 설정은 처음부터 동시 학습이며, 기존 SITT/VAIC checkpoint를 CTS checkpoint로
로드하지 않는다.

## 네트워크와 objective

CTS의 학습 논리를 기존 VAIC perception architecture에 적용한다.
원문의 blind-locomotion MLP를 그대로 복제하는 구현은 아니다.

- Teacher: 기존 `priv + object_ + object_trans -> encoder_priv`.
  Task에 height가 있을 때만 기존 height CNN도 사용한다.
- Student: 기존 `depth CNN + TemporalDepthGRU -> object_adapt ->
  predicted object transform -> adapt_module GRU` 전체를 사용한다.
  `object_geo_`, 기존 policy observation과 전체 `vel_command` 입력을 유지한다.
- Actor: 기존 `actor_adapt`의 `[vel_command, policy, latent]` 입력과 layer 구조를
  양 그룹이 공유한다. Reference joint-position residual은 적용하지 않는다.
- Critic: 기존 privileged critic observation에 선택한 latent를 추가한다.
  Reward-group별 value head와 기존 task reward aggregation은 유지한다.
- Latent: 기존 256차원을 유지하고 CTS 원문처럼 양 encoder 출력에 L2 normalization을
  적용한다. `algo.cts.normalize_latent=false`는 별도 ablation이며 학습/평가 시 일치해야 한다.

한 iteration은 다음 순서다.

1. 두 그룹이 각자의 encoder와 공통 actor로 stochastic rollout을 수집한다.
   Student rollout은 학습 중인 **online perception**을 사용한다.
2. 각 실제 trajectory의 reward와 privileged critic으로 GAE를 계산한다.
   Reward-group 가중합 후 teacher/student 내부에서 각각 advantage를 정규화한다.
3. `L_PPO_teacher.mean() + L_PPO_student.mean()`으로 shared actor를 업데이트한다.
   Teacher PPO만 privileged encoder를 업데이트한다. Student latent는 PPO epoch 동안
   저장된 rollout 값을 사용하므로 student perception에는 PPO gradient가 들어가지 않는다.
   Value loss는 두 그룹 전체의 평균이며 critic 입력 latent는 detach한다.
4. Student 그룹의 observation sequence를 재생한다. **같은 student 환경, 같은 시점**의
   privileged observation을 업데이트된 teacher encoder에 넣어 detached target을 만든다.
   `mean(||z_student - stopgrad(z_teacher)||²) + object_loss_coef * object_MSE`로
   depth CNN/temporal GRU, object predictor, adaptation GRU만 업데이트한다.

Object MSE는 VAIC perception의 기존 보조 학습을 보존한 항이다(기본 계수 1).
Student action imitation, proxy, teacher-student KL reward/regularization은 사용하지 않는다.
Task observation 생성, noise/delay, camera DR, reward, termination, action 설정은
`vaic/skateboard_general_tracking_stu.yaml`과 같다. 모든 환경의 depth를 생성하며
selective rendering이나 image replay buffer는 추가하지 않는다.

Teacher-only inference와 student-only inference 모두 지원하지만, 평가와 배포의 기본은
student다. 학습 rollout과 평가 모두 동일한 online perception을 사용한다.
SITT/VAIC의 EMA perception을 평가에서 바꿔 끼우지 않는다.
`get_rollout_policy("deploy")`는 student perception과 shared actor만 포함하는 독립 모듈을
반환한다. Teacher/critic 없이 복사·CPU 이동·PyTorch 직렬화가 가능하다. 입력에는
checkpoint에 저장한 observation normalizer를 적용해야 한다.

## W&B: 기존 VAIC key = student, `cts_teacher/*` = teacher

| Key | 데이터 출처 |
| --- | --- |
| `reward.<group>/<term>` | Student 환경의 term별 reward EMA (기존 VAIC 방식) |
| `train/stats/success`, `train/stats/episode_len` | 종료된 student episode 평균 |
| `train/stats/<group>/<term>` | 종료된 student episode의 누적 term 통계 |
| `actor/policy_loss`, `actor/kl`, `critic/*.value_loss` | Student rollout sample의 최적화 진단 |
| `adapt/priv_loss`, `adapt/object_loss` | Student trajectory의 latent/object 원소별 MSE |
| `eval/*`, `eval/*_std` | 학습 종료 후 모든 환경을 student로 실행한 평가 |
| `cts_teacher/reward/<group>/<term>` | Teacher 환경만의 term별 reward EMA |
| `cts_teacher/episode/success`, `cts_teacher/episode/episode_len` | 종료된 teacher episode 평균 |
| `cts_teacher/episode/<group>/<term>` | 종료된 teacher episode의 누적 term 통계 |
| `env_frames`, `rollout_fps`, `training_time`, `performance*`, `iterations` | 전체 학습 처리량/시간 (student 성능 지표가 아님) |
| `actor/lr`, `actor/entropy_coef`, `actor_std/*` | 공유 actor의 설정/파라미터 |

Teacher-only episode가 끝나도 student episode 통계에 포함하지 않는다.
Episode key는 해당 그룹의 종료 episode가 있을 때만 기록된다. VAIC와 같은
logging interval로 집계하며 마지막 iteration에도 남은 episode 통계를 기록한다.
Episode term 값은 누적 합이다. Reward EMA는 그룹별 weighted reward 합과
active sample 수를 각각 decay=0.99로 누적한다. Active sample이 한 번도 없었던
그룹/term은 정의되지 않은 평균 대신 key를 생략한다. `metrics.py`에서 기존 reward
호출 결과를 관찰할 뿐 재계산하거나 reward/학습 objective를 수정하지 않는다.
이미 두 그룹이 합쳐진 전역 reward EMA와 `env.extra`는 전달하지 않는다.
현재 task의 `env.extra`는 비어 있으며, 전역 simulator timing만 기존 key로 유지한다.
`env_frames`는 VAIC와 동일하게 teacher+student 전체 환경 step 수이다.
`adapt/priv_loss`는 비교용 원소별 MSE이며 실제 CTS latent objective는 기존대로
squared L2 합이다. CTS latent 정규화 등 알고리즘 차이가 있으므로 loss의 절댓값이
다른 알고리즘과 완전히 같은 의미는 아니다. `student/*` key는 더 이상 생성하지 않는다.

최종 평가는 deterministic student로만 수행한다. 각 환경의 첫 종료 episode를 한 번씩
집계하고, 미종료 환경은 `eval/incomplete_episodes`로 별도 보고한다. `episode_cnt`는
집계된 첫 완료 episode 수이다. VAIC evaluator와 동일하게 success/episode_len은
그대로 평균하고 나머지 term은 각 episode 길이로 나눈 후 평균/표준편차를 기록한다
(termination term도 포함). 평가 시 observation normalizer를 freeze한다.
전체 학습이 끝나기 전의 성능은 `train/stats/*`로 확인한다.
Teacher rollout 지표는 별도로 기록하지만 최종 teacher 평가를 추가 실행하지는 않는다.
기존 W&B run의 과거 key는 자동으로 변경되지 않으며 수정 후 실행부터 적용된다.

## 평가와 재개

Student 시각화 (`cfg/play.yaml`의 기본 환경 수는 16):

```bash
python scripts/play.py \
  algo=cts_ppo_vel_train \
  task=G1/cts/skateboard_general_tracking \
  checkpoint_path=/absolute/path/checkpoint_300.pt
```

Teacher 비교 시에는 `algo.cts.eval_policy=teacher`를 명시한다.
Training runner의 최종 W&B 평가는 이 값과 관계없이 student-only로 수행한다.

CTS 학습 재개:

```bash
python scripts/train.py \
  algo=cts_ppo_vel_train \
  task=G1/cts/skateboard_general_tracking \
  checkpoint_path=/absolute/path/checkpoint_300.pt
```

`run:<entity>/<project>/<run_id>`도 지원한다. `total_frames`는 추가 frame 수가 아닌
전체 teacher+student 누적 budget이다. 재개 시 `num_envs`, `train_every`, CTS 학습 설정이
저장본과 같아야 한다. 모델, optimizer, normalizer, LR, iteration, RNG를 복원하고
simulator는 새로운 episode로 시작한다. `eval_policy`/`evaluate_at_end`는 재개 시 변경 가능하다.

원문 비율 teacher:student=3:1을 사용하려면 `algo.cts.teacher_ratio=0.75`로 설정한다.
총 1024개일 때 768/256이며, 각 그룹의 환경 수는 `algo.num_minibatches`로 나누어져야 한다.
현재 지원 범위는 단일 GPU/process이다.

## 코드 경계와 검증

`config.py`, `algorithm.py`, `networks.py`, `runner.py`는 CTS 전용이다.
SITT 모듈을 import하지 않는다. 기존 `ppo_vel.py`, SITT, VAIC task, 환경/reward 구현은
수정하지 않으며 공통 진입점에는 CTS 등록 import와 runner dispatch만 추가한다.

```bash
python -m unittest discover -s tests/cts -v
CTS_TEST_DEVICE=cuda python -m unittest discover -s tests/cts -v

python scripts/train.py algo=cts_ppo_vel_train task=G1/cts/skateboard_general_tracking \
  task.num_envs=8 algo.train_every=16 algo.num_minibatches=2 algo.ppo_epochs=1 \
  algo.cts.reconstruction_epochs=1 algo.cts.evaluate_at_end=false \
  total_frames=256 save_interval=1 wandb.mode=disabled
```

Tests cover rollout selection, student privileged-input isolation, gradient boundaries,
paired reconstruction targets, group advantage normalization, terminal bootstrap,
checkpoint/optimizer continuation, single-env inference, student-only logs and task parity.

참고: [CTS 원문 §II, Algorithm 1](https://arxiv.org/html/2405.10830v2),
[저자 프로젝트](https://clearlab-sustech.github.io/concurrentTS/),
[제3자 PPO 구현](https://github.com/lupinjia/LeggedGym-Ex/blob/main/rsl_rl/algorithms/ppo_cts.py).
제3자 코드를 저자의 공식 구현으로 취급하지 않으며 latent normalization과 critic의
latent 입력은 원문을 기준으로 적용했다.
