# VAIC용 Student-Informed Teacher Training

실행:

```bash
conda activate vaic
python scripts/train.py algo=sitt_ppo_vel_train task=G1/sitt/skateboard
```

Mirror motion을 포함한 general-tracking task:

```bash
python scripts/train.py algo=sitt_ppo_vel_train task=G1/sitt/skateboard_general_tracking
```

SITT는 논문의 표기이며, 이 저장소에서는 CLI/package 표기로 `sitt`를 사용한다.
본 구현은 **VAIC의 observation, perception encoder와 환경 reward를 보존하면서**
논문과 같이 teacher, student, proxy가 하나의 action decoder를 공유한다.

## 코드 경계

| 파일 | 역할 |
| --- | --- |
| `config.py` | `sitt_ppo_vel_train` Hydra 등록 및 SITT 설정 |
| `algorithm.py` | 기존 `PPOVEL` 네트워크를 재사용하고 SITT 업데이트만 구현 |
| `networks.py` | 학습 전용 proxy student |
| `losses.py` | detached-target KL, warm-up/ramp, decoder freeze |
| `runner.py` | teacher rollout, recurrent state 수집, bootstrap, 저장 및 평가 |
| `cfg/task/G1/sitt/skateboard*.yaml` (저장소 루트 기준) | 기본 및 general-tracking SITT task |
| `tests/sitt/test_sitt.py` (저장소 루트 기준) | 실제 PyTorch/TensorDict 모듈 검증 |

공통 파일의 변경은 `learning/__init__.py`의 설정 등록 import와 `scripts/train.py`의
SITT runner 선택 및 현재 checkout import 보장이다. 기존 `ppo_vel.py`, 환경,
reward 구현, `cfg/task/G1/vaic/*`는 수정하지 않는다.

기존 skateboard YAML의 상대 defaults 경로(`base/hdmi-base`)는 현재 위치에서
Hydra가 해석하면 존재하지 않는 `task/G1/vaic/base/hdmi-base`를 가리킨다.
따라서 SITT task는 student task 내용을 독립적으로 보존하고 절대 Hydra 경로
`/task/base/hdmi-base`를 사용한다. 테스트가 양 task의 observation, reward,
action, command, termination, randomization, camera 설정 동등성을 확인한다.

## 보존하는 VAIC 경로

Teacher encoder:

```text
priv + GT object/object transform [+ optional height encoder]
    → encoder_priv → priv_feature
```

현재 skateboard task에는 `height` observation이 없다. 단순히 코드에 height CNN이
존재한다는 이유로 height 입력을 추가하지 않는다.

Student encoder:

```text
depth → 기존 CNN + TemporalDepthGRU → depth feature
policy + vel_command + depth feature → object_adapt → object_pred
object_pred + object_geo_ → predicted object transform
policy + vel_command + predicted object/transform → adapt_module GRU → priv_pred
```

Shared action decoder:

```text
teacher: vel_command + policy + priv_feature   ┐
student: vel_command + policy + priv_pred      ├→ shared_actor → action distribution
proxy:   vel_command + policy + proxy_feature  ┘
```

`vel_command`는 기존 YAML에 정의된 future velocity/contact 입력 전체를 유지한다.
3차원 command로 축소하지 않는다. `object_geo_`도 원래 student가 사용하던 입력이다.
Teacher의 motion command, GT object state, privileged observation 및 reference joint
position은 shared decoder 입력에 전달되지 않는다. Teacher와 student의 observation
생성 함수, history, camera noise/latency/DR, CNN/GRU 구조 및 latent 차원은 유지한다.
기존 teacher 전용 actor와 `ref_joint_pos_` residual decoder는 SITT policy에서 사용하지
않는다. Teacher의 privileged 정보는 `encoder_priv`를 통해 공통 latent 공간으로 들어간다.

기존 `PPOVEL(phase=train)`은 adaptation GRU를 GT object에 연결한다. 이 새 클래스는
GRU의 layer/크기/초기 가중치를 유지하면서 입력 key를 기존 `phase=finetune`과 같은
`object_pred`, `object_pred_trans`로 구성한다. Student는 처음부터 이 경로를 학습한다.

Proxy encoder는 student perception의 복사본이 아닌 **별도 MLP**다:

```text
priv + object_ + object_trans
    → teacher encoder와 같은 순서로 concat
    → Linear(256) + LayerNorm + Mish
    → Linear(256) → proxy_feature
```

이는 현재 skateboard의 `encoder_priv`와 정확히 같은 input contract와 MLP layer
구조다. 두 encoder의 파라미터는 공유하거나 복사하지 않고 각각 초기화하고 학습한다.
`object_trans`는 두 경로 모두 기존 `object_transform(object_, object_geo_)`로 만든다.
같은 simulation step의 TensorDict에서 읽으며 학습 중 정규화 대상 key는 VecNorm이
처리한 값이다. `priv`에는 깨끗한 proprioception history, reference tracking 정보와
object 상태 등이 포함된다. `vel_command`와 `policy`는 encoder에 중복 입력하지 않고
teacher/proxy 모두 shared decoder에서 공통으로 받는다. Proxy에는 CNN, object predictor,
GRU hidden state가 없다.

GT object는 proxy의 **입력**이다. Target은 GT object나 teacher latent가 아니라,
현재 실제 student의 `priv_pred`와 decoded action이다. 따라서 proxy는 student의
현재 perception 오차와 관측 한계를 포함한 출력을 근사하도록 학습한다. Object MSE는
student에만 적용하고 proxy에는 feature/action L1을 적용한다. Encoder 파라미터는
서로 독립이며 공유하는 것은 action decoder와 std뿐이다.

논문 §5의 vision 실험에서도 teacher/proxy는 같은 MLP 계열이고 student는 image
encoder+MLP이므로 student와 proxy의 구조는 다르다. 이 구현은 teacher/proxy MLP
구조를 맞추면서 student CNN/GRU와 shared decoder는 기존 VAIC 구조를 사용한다.

## 한 iteration의 목적함수와 업데이트

1. **Teacher rollout**: teacher만 action을 샘플링하고 환경을 움직인다. 같은 state에서
   실제 student의 depth/adaptation GRU hidden state를 진행시킨다. Proxy는 raw privileged
   observations로 student latent를 예측하고 shared actor로 이를 decode한다.
   Teacher의 action, loc/scale, log-prob와 proxy loc/scale을 별도 key로 고정 저장한다.
2. **Teacher PPO**: 기존 task reward group critic을 유지하고, 별도 `sitt_critic`으로
   `-KL`의 return을 학습한다. 환경 reward tensor는 변경하지 않는다.
3. **Alignment phase, student step**: 업데이트된 teacher를 frozen target으로 사용한다. 기존 VAIC의
   object MSE를 유지하고, 논문처럼 feature L1과 frozen shared-decoder action L1을 함께
   사용해 student perception encoder만 갱신한다.
   아래 proxy step과 같은 minibatch에서 순서대로 실행한다.
4. **같은 alignment phase, proxy step**: 방금 갱신된 실제 student를 재계산해 frozen target으로 사용한다. Latent L1과
   action L1으로 proxy만 학습한다. 이때 shared actor는 가중치를 freeze하되 proxy
   latent에 대한 미분은 허용한다. 마지막으로 student perception EMA를 갱신한다.

`d_t = KL(teacher || proxy)`는 `IndependentNormal`의 **action 차원 합**이다.
기본값은 clipping이나 action-dimension averaging 없는 Gaussian KL이다.

```text
r_KL,t = -step_dt * d_t                 # kl_reward_mult_dt=true
A_KL = GAE(r_KL, V_KL)
A_teacher = normalize(sum_g w_g * A_task,g + λ_reward * A_KL)
L_teacher = L_PPO + L_task_value + L_KL_value - entropy_coef * H
            + λ_policy * mean KL(teacher || frozen proxy)
```

기존 task reward는 이미 `step_dt`를 곱하므로 KL reward도 같은 시간 단위를 사용한다.
`kl_reward_mult_dt=false`이면 이 추가 scaling을 끌 수 있다. 직접 policy KL loss에는
`step_dt`를 곱하지 않는다. 두 λ는 독립적이며 실제 skateboard 결과로 조정해야 한다.

GAE의 선형성 때문에 위 분리는 기존 task reward의 가중 합에 `-λ_reward * step_dt * KL`을
더한 단일 reward의 GAE와 대응한다. 기존 reward group 개수/weight와 task critic의
출력을 바꾸지 않고 task return을 그대로 기록할 수 있다. KL을 모두 끄면 teacher의
advantage와 task return이 기존 코드와 일치하는지 테스트한다. Decoder 입력 구조가
다르므로 기존 residual teacher와 파라미터 업데이트 자체가 같다는 의미는 아니다.

Shared mean decoder는 teacher PPO에서 갱신된다. 이 PPO advantage에는 task reward와
KL penalty가 모두 반영된다. 직접 KL loss에서는 mean decoder를 freeze하되 teacher
encoder와 공통 state-independent std에는 gradient를 보낸다. 매 PPO minibatch마다
현재 frozen proxy encoder와 **현재 공유 decoder**로 proxy mean을 다시 계산하여
detach한다. Rollout에 저장한 proxy loc/scale은 직접 KL target으로 재사용하지 않는다.
Student/proxy encoder에는 teacher PPO gradient가 흐르지 않는다.

연속 action의 직접 KL은 논문 식 (10)의 공통 covariance 식을 사용한다:

```text
L_KL = 0.5 * sum_j ((mu_teacher,j - stopgrad(mu_proxy,j)) / sigma_j)^2
d L_KL / d sigma_j = -(mu_teacher,j - mu_proxy,j)^2 / sigma_j^3
```

논문 §4.1의 decoder freeze와 §4.2의 covariance 증가 설명을 함께 반영하기 위해
mean decoder와 std의 gradient를 구분한다. 공통 std까지 detach하면 식 (10)의
covariance gradient가 사라진다. Alignment에서는 std도 포함해 전체 decoder를 동결한다.

Alignment의 feature L1 + decoder output L1은 공개 toy 코드 `align_policy`를 따른다.
논문 §5의 decoder는 단일 FC이므로 원 논문의 decoder activation은 그 출력이다.
VAIC의 다층 decoder를 보존하며, 논문에 없는 중간 layer별 추가 L1 항은 도입하지 않는다.

Student에 teacher 행동 데이터를 이용한 PPO를 적용하지 않는다. 기존 두 번째 명령의
student on-policy finetuning은 이 SITT loop에 포함되지 않는다. 여기서는 student를
동시에 증류하는 것으로 대체한다. 따라서 기존 two-stage 결과와 성능 동등성을 보장하지
않으며, teacher/student 각각의 task return과 실제 배포 성능으로 비교해야 한다.

## 초기 학습과 proxy 한계

기본값은 100 iteration warm-up 후 200 iteration에 걸친 λ ramp이다.
Warm-up 중에도 teacher PPO와 student/proxy alignment는 모두 진행된다.
이 시간 스케줄은 proxy 수렴을 자동 판정하지 않는다. `proxy/feature_loss`,
`proxy/action_loss`, `sitt/rollout_kl`을 보고 ramp 길이를 조정한다.

논문은 비용 절감을 위해 일부 state/environment에서만 이미지를 생성한다.
현재 구현은 기존 camera 시스템을 유지하기 위해 모든 설정된 환경에서 depth를 생성한다.
따라서 proxy를 사용하더라도 원 논문의 selective rendering 속도 이득은 구현하지 않는다.
History-dependent student를 state-based proxy가 완전히 복원할 수도 없으며, proxy
오차는 학습 결과와 함께 확인해야 한다. 무작위 image subsampling은 기존 GRU의 시간
간격/관측 history를 바꿀 수 있으므로 적용하지 않는다.

기존 student rollout의 perception EMA는 그대로 유지한다. Training rollout에서
student history/target은 online perception을 사용하며, 평가/배포에는 EMA perception을
사용한다. Action decoder와 std는 처음부터 하나의 module이므로 별도 동기화가 없다.

정규화는 기존 VecNorm 구현을 사용하되, joint training 중에는 두 관측 경로의 통계를
함께 학습한다. 이전 teacher checkpoint의 frozen 통계를 사용하는 two-stage finetune과
정규화 스케줄은 다르다. 저장 시 VecNorm을 반드시 포함하고 평가에서는 freeze한다.

## 실행, 검증 및 재개

작은 simulation smoke run:

```bash
python scripts/train.py algo=sitt_ppo_vel_train task=G1/sitt/skateboard \
  task.num_envs=8 algo.train_every=16 algo.num_minibatches=2 algo.ppo_epochs=1 \
  algo.sitt.alignment_epochs=1 algo.sitt.proxy_epochs=1 \
  algo.sitt.warmup_iters=0 algo.sitt.ramp_iters=0 \
  algo.sitt.evaluate_at_end=false total_frames=256 save_interval=0 wandb.mode=disabled
```

CPU tests (Isaac Sim 실행 없이 실제 VAIC layer를 테스트):

```bash
python -m unittest discover -s tests/sitt -v
```

현재 SITT task의 기본 환경 수 1024와 rollout 길이 32를 사용한다. 현재 지원 범위는
단일 GPU, 원래 depth/object/GRU 구조와 기본 advantage aggregation이다. 지원하지 않는
ablation/distributed 설정은 조용히 다른 학습을 수행하는 대신 초기화에서 오류를 낸다.

재개는 새 SITT checkpoint만 받는다:

```bash
python scripts/train.py algo=sitt_ppo_vel_train task=G1/sitt/skateboard \
  checkpoint_path=/absolute/path/checkpoint_final.pt total_frames=1000000000
```

명칭 통일 이전 또는 독립 teacher/student actor로 생성된 version-1 checkpoint와 기존
shared-decoder version-2 checkpoint는 받지 않는다. Version 3에서 proxy를 teacher와
같은 input contract/MLP 구조로 변경했기 때문에 module shape과 optimizer state가 다르다.
새 checkpoint에는 `sitt_version=3`, `sitt_objective_version=2`를 기록한다.

`run:<entity>/<project>/<run_id>`도 지원한다. `total_frames`는 추가 frame 수가 아니라
최종 누적 budget이다. Resume 시 `num_envs`와 `train_every`는 저장 당시와 같아야 한다.
모델, EMA, proxy, 두 critic, optimizer, iteration/LR, normalizer와 RNG를 복원한다.
물리 simulator state와 episode history는 직렬화하지 않으므로 새로운 episode에서
계속하며, 중단 직전 trajectory를 bitwise 재현하는 기능은 아니다. 저장 함수 자체는
student/EMA 파라미터를 수정하지 않는다. 기존 teacher/finetune checkpoint를 silent
partial load하지 않는다.

학습 종료 시 teacher와 EMA student를 각각 reset하여 별도로 평가한다.
`get_rollout_policy("eval")`의 기본은 student이며, `algo.sitt.eval_policy=teacher`로
선택할 수 있다. `get_rollout_policy("deploy")`는 항상 student 경로를 반환한다.
기존 evaluation scripts를 사용할 때 다른 checkout의 editable install을 피하려면
저장소 루트에서 `PYTHONPATH=. python scripts/eval.py ...`로 실행한다.

## 검증 기록

2026-09-14, `vaic` conda 환경(PyTorch 2.7.0+cu128, TorchRL/TensorDict 0.7.0),
RTX 5090에서 다음 검증을 완료했다.

- CPU unittest 16개 통과: 기본/general-tracking task 설정 일치, KL 부호/GAE,
  recurrent reset, 단일 shared decoder, optimizer/gradient 분리, current-proxy KL,
  공통 covariance의 값/gradient, action loss의 전체 perception gradient,
  배치별 paired alignment, checkpoint 복원 및 collector bootstrap 등을 확인했다.
- 이전 shared-decoder 초기 버전의 GPU smoke train(8 env, rollout 16, 2 iteration)에서 version-2 checkpoint에
  action module이 `shared_actor` 하나만 저장되고, policy/두 critic/student/proxy의
  다섯 optimizer가 모두 갱신되며 모든 저장 tensor가 finite임을 확인했다.
- 이 checkpoint를 다시 로드해 1 iteration을 추가 학습한 뒤 teacher와 EMA student
  평가 경로를 각각 32 step 실행했다. 재저장된 checkpoint는 누적 update 3회와 각
  optimizer step 6회를 보존했다.
- 이번 목적함수 수정 후 CUDA 모델 검증: synthetic observation/transition으로 3회
  학습, teacher/student inference, checkpoint state 복원을 완료했다. 복원 직후 가중치는
  정확히 같았고, 동일 데이터로 다음 업데이트를 수행한 뒤 최대 파라미터 차이는
  `1.86e-9`였다. PPO/두 critic optimizer는 16 step, student/proxy는 8 step까지 진행했다.
  이 검사는 물리 simulation 성능 검증이 아니다.
- Version-3 proxy 구조에서도 CUDA update와 checkpoint 복원을 다시 실행했다.
  Proxy input은 `priv/object_/object_trans`, teacher/proxy parameter 공유는 없으며
  update 결과와 복원된 모든 parameter가 finite임을 확인했다.
- 이번 Isaac Sim smoke 실행은 다른 GPU 작업이 메모리를 사용 중인 상태에서 PhysX의
  약 2.12 GB 할당 실패로 scene 생성 단계에서 중단됐다. 따라서 수정 후 simulator
  학습/평가가 성공했다고 주장하지 않는다. 로그:
  `/tmp/vaic-sitt-current-proxy.bZi0L5/smoke.log`.

이는 실행 경로와 짧은 수치 안정성 검증이며, 장기 수렴·보행/스케이트보드 성능이나
최적 KL 계수를 검증한 실험은 아니다. 설치된 TorchRL C++ extension 호환성 경고와
이전 Isaac Sim 종료 시 semaphore 경고도 관찰됐다.

## 참고 및 재현 차이

- [공식 ICLR 2025 논문, §3–4 및 Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf)
- [공식 code repository](https://github.com/uzh-rpg/sitt)
- 로컬 참고 코드: `/home/hcc/research/sitt/color_maze/rl_scripts/`

공개 toy 코드는 policy update에서 feature L1을 사용하며 연속 covariance 구현을
제공하지 않는다. 본 구현은 연속 action에 대해 논문 식 (10)을 따른다.
공개 toy 코드의 `rewards += 1.9 * KL`은 논문 Algorithm 1의 `r - λ * KL`과 부호가
다르다. 본 구현은 논문의 **음수 penalty**를 따른다. VAIC의 중간 object predictor에는
기존 object MSE를 보조 loss로 유지한다. 전체 camera rendering과 online/EMA student
구분은 VAIC 구조를 보존하기 위한 차이이며, 기본 계수는 검증된 최적값이 아니다.
