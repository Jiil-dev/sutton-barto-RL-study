# Chapter 1 (틱택토) 사용 설명서

> Sutton & Barto, *RL: An Introduction* (2nd Ed.) Chapter 1을 따라 만든
> 틱택토 RL 학습/실험 코드의 통합 매뉴얼.
>
> 본 문서는 다음을 다룹니다.
> 1. 프로젝트 구조와 모듈 책임
> 2. 빠른 시작 (실행/검증 방법)
> 3. 모듈별 사용법과 인터페이스
> 4. 핵심 설계 결정과 그 이유
> 5. 유의할 점 (실수하기 쉬운 부분)
> 6. 가설 검증 결과 요약
> 7. 확장 방향

---

## 1. 프로젝트 구조

```
sutton-barto-RL-study/
├── README.md                       # 진행 현황, 자료 링크, 결과 요약
├── reqiurements.txt                # numpy, matplotlib, tqdm
├── code/ch01/
│   ├── CLAUDE.md                   # 자율 작업 지시서 (Claude Code용)
│   ├── MANUAL.md                   # ← 이 문서
│   ├── env.py                      # 틱택토 환경 (Gym-style)
│   ├── train.py                    # 학습 루프 (play_game / train / evaluate)
│   ├── utils.py                    # 시각화/저장/시드/대칭(D4) 헬퍼
│   ├── agents/
│   │   ├── random_agent.py         # 무작위 베이스라인
│   │   ├── td_agent.py             # TD(0) afterstate 학습
│   │   └── q_agent.py              # Q-learning 확장
│   └── experiments/
│       ├── ex1_self_play.py        # Ex 1.1 자가 대국
│       ├── ex2_symmetries.py       # Ex 1.2 대칭성
│       ├── ex3_greedy_only.py      # Ex 1.3 탐험 vs 탐욕
│       ├── ex4_learn_from_exploration.py  # Ex 1.4 on/off-policy
│       └── ex5_q_learning.py       # Ex 1.5 Q-learning
├── notes/ch01/
│   ├── 1.5_tic_tac_toe.md          # 본문 번역 + 보충
│   ├── exercises.md                # Ex 1.1~1.5 답변과 가설 (H1.xa, H1.xb, ...)
│   ├── implementation_notes.md     # 환경/에이전트 설계 결정 기록
│   └── progress_log.md             # 코드 작성 진행 로그
└── results/ch01/                   # 실험 산출 PNG (16개)
```

### 의존 그래프 (개념적)

```
env.py (독립)
   ↑
   ├── agents/random_agent.py   (env import 없이도 동작; 인터페이스만 맞춤)
   ├── agents/td_agent.py        (TYPE_CHECKING으로만 env 참조)
   └── agents/q_agent.py
   ↑
train.py  ← play_game / train / evaluate 모두 여기.
   ↑
utils.py  ← 결과 시각화, 저장, 대칭. (env 상수만 사용)
   ↑
experiments/ex*.py  ← 위의 모든 모듈 조합.
```

---

## 2. 빠른 시작

### 2-1. 환경 준비

```bash
git clone https://github.com/Jiil-dev/sutton-barto-RL-study.git
cd sutton-barto-RL-study
python3 -m venv venv
source venv/bin/activate
pip install -r reqiurements.txt
```

> ⚠️ 파일명이 `requirements.txt`가 아니라 `reqiurements.txt` (오타). 그대로 사용.

### 2-2. 모든 모듈이 정상 동작하는지 빠르게 확인

각 파일은 `if __name__ == "__main__":` 블록에 self-test가 들어 있습니다.

```bash
cd code/ch01
python train.py                 # play_game / train / evaluate 7개 테스트
python utils.py                 # 저장·시각화·대칭 7개 테스트
python agents/random_agent.py   # 무작위 정책 5개 테스트
python agents/q_agent.py        # Q-learning 6개 테스트
```

각각 `✅ 모든 테스트 통과` 출력이 나오면 OK.

> ⚠️ `python env.py`는 사람 vs 사람 대화형 게임이 시작됩니다 (입력 대기). 자동 검증용이 아닙니다.
>
> ⚠️ `python agents/td_agent.py`의 self-test는 시드를 고정하지 않고 하드코딩된 `legal_actions=[0,1,2,3]`을 쓰는 **사전 존재 결함**이 있어 가끔 실패합니다 (코드 본체는 정상). 본체 검증은 `train.py` 자가 테스트 [5]번 또는 ex1/ex3/ex4 실행으로 대신할 수 있습니다.

### 2-3. 실험 5개 실행

```bash
cd code/ch01
python experiments/ex1_self_play.py
python experiments/ex2_symmetries.py
python experiments/ex3_greedy_only.py
python experiments/ex4_learn_from_exploration.py
python experiments/ex5_q_learning.py
```

각 실험은:
- 콘솔에 가설별 결과(`✅ 지지`/`⚠️ 재검토` 등)를 출력하고
- `results/ch01/ex{N}_*.png`에 그래프를 저장합니다.

전체 실행 시간은 5분 내외 (i7급 CPU 기준, 시드 고정).

---

## 3. 모듈별 사용법과 인터페이스

### 3-1. `env.py` — 틱택토 환경

Gym-style 인터페이스. 상태는 길이 9 튜플로 `(EMPTY=0, PLAYER_X=+1, PLAYER_O=-1)`.

```python
from env import TicTacToeEnv, PLAYER_X, PLAYER_O

env = TicTacToeEnv()
state = env.reset()                          # 빈 보드 상태
while not env.done:
    action = ...                              # 0~8 중 합법 수
    state, reward, done, info = env.step(action)
```

**핵심 메서드**
| 메서드/속성 | 반환 | 비고 |
|---|---|---|
| `reset()` | `State` | 게임 초기화 후 빈 보드 |
| `step(action)` | `(State, float, bool, dict)` | 보상은 둔 사람 시점 (승+1/패-1/무0/진행0) |
| `legal_actions()` | `list[int]` | 빈 칸 인덱스 |
| `simulate_step(action)` | `State` | **실제 변경 없이** 가상 afterstate 반환 |
| `current_player` | `int` | `@property`. 종료 시에도 마지막에 둔 사람으로 유지 |
| `winner` | `int \| None` | None이면 무승부 또는 진행 중 |
| `done` | `bool` | `@property` |
| `render()` | `str` | 디버깅용 문자열 보드 |

### 3-2. `agents/*.py` — 에이전트 공통 인터페이스

세 에이전트 모두 동일한 인터페이스를 만족하므로 `train.py`가 분기 없이 호출 가능.

```python
class Agent(Protocol):
    training: bool                          # 학습/평가 모드 토글

    def choose_action(self, state, legal_actions, env) -> tuple[int, bool]:
        """반환: (action, was_exploratory)"""

    def observe(self, prev_afterstate, curr_afterstate,
                reward, done, was_exploratory) -> None:
        """학습 갱신. 학습 안 하는 에이전트는 no-op."""
```

#### `RandomAgent`
- `choose_action`: 합법 수 중 균등 무작위. `was_exploratory`는 항상 False (탐욕 정책 미정의).
- `observe`: no-op.
- `training` 토글은 placeholder (동작 영향 없음).

#### `TDAgent`
- TD(0) afterstate 학습. `V: dict[State, float]`.
- 갱신 규칙: `V(prev) ← V(prev) + α(V(curr) - V(prev))` (탐욕 수에서만, 종료 시엔 `V(curr)=reward`).
- 주요 인자: `alpha`, `epsilon`, `default_value=0.0`, `learn_from_exploration=False`.
- `learn_from_exploration=True`로 켜면 탐험 수에서도 갱신 (Ex 1.4 on-policy 비교용).

#### `QAgent`
- 표준 Q-learning. `Q: dict[(State, int), float]`.
- 갱신 규칙: `Q(s,a) ← Q(s,a) + α(r + γ·max_a' Q(s',a') - Q(s,a))`.
- γ=1, 종료 시 bootstrap 없음.
- 인자: `alpha`, `epsilon`, `gamma=1.0`, `default_value=0.0`.
- ⚠️ `prev_afterstate`/`curr_afterstate` 인자는 받지만 **사용하지 않음**. 내부 `_pending_state/action`로 (s,a) 추적.

### 3-3. `train.py` — 학습 루프

```python
from train import play_game, train, evaluate

# 한 게임만 돌리기
winner = play_game(env, agent_x, agent_o, learn=True)

# N판 학습 + 주기적 평가 (학습 곡선 history 반환)
history = train(
    agent_x, agent_o,
    num_games=5000,
    eval_every=250,                                # 250판마다
    eval_games=300,                                # 평가 1회당 300판
    eval_opponent_factory=lambda: RandomAgent(),   # 매번 새 인스턴스
    eval_target="x",                               # "x" 또는 "o"
)
# history = {"results": [...], "eval_steps": [...], "eval_records": [...]}

# 단발 평가
rec = evaluate(agent, lambda: RandomAgent(), num_games=1000)
# rec = {"wins": ..., "losses": ..., "draws": ...}  (합 = 1.0)
```

### 3-4. `utils.py` — 시각화/저장/대칭

```python
from utils import (
    set_seed, ensure_results_dir,
    save_values, load_values,
    plot_learning_curves, plot_outcome_distribution, plot_first_move_heatmap,
    first_move_distribution, measure_game_length,
    SYMMETRIES, apply_symmetry, canonical_state,
)

set_seed(42)                          # random.seed(42) 동등
results_dir = ensure_results_dir()    # results/ch01/ 생성

save_values(agent.values, "v.pkl")    # pickle
v = load_values("v.pkl")

# 학습 곡선
plot_learning_curves(
    {"label1": history1, "label2": history2},
    title="...",
    save_path=results_dir / "ex1.png",
    metric="wins",
)

# 정책 분석
dist = first_move_distribution(agent, env, n_samples=2000)  # {0: p0, 1: p1, ..., 8: p8}
plot_first_move_heatmap(dist, "title", path)

# 대칭 (D4 group, 8개)
canonical = canonical_state(state)    # 8개 변환 중 사전식 최소
```

### 3-5. `experiments/ex*.py` — Exercise 검증 스크립트

각 스크립트는 다음 형식을 따름.
1. 상단 docstring에 검증 가설 열거
2. 하이퍼파라미터 상수 정의 (`NUM_GAMES`, `ALPHA`, `EPSILON`, `SEED`, …)
3. `main()` 함수에서 학습 → 평가 → 그래프 저장 → 가설별 판정 로그 출력

이 형식 덕분에 **하이퍼파라미터를 바꿔 재실험**하기 쉽습니다 (스크립트 상단 상수만 수정).

---

## 4. 핵심 설계 결정과 그 이유

### 4-1. 상태 표현: `tuple[int, ...]`

> 후보: numpy.array (빠르지만 unhashable), tuple (hashable + immutable), str (변환 비용)

선택 이유: 가치 함수가 `dict[State, float]`라 키는 hashable이어야 함. 9칸이라 tuple로 충분히 빠름. 또한 immutable이라 실수로 변경될 위험 차단.

### 4-2. 플레이어를 `+1/-1`로 인코딩

`PLAYER_X=+1, PLAYER_O=-1, EMPTY=0`로 두면 `-player`로 시점 전환이 가능해, 자가 대국 코드가 깔끔해집니다 (보상 부호만 뒤집으면 양쪽이 같은 코드 경로 사용).

### 4-3. 보상 규약 `+1/-1/0`

책의 §1.5는 0/1/0이지만, 본 프로젝트는 부호 반전 활용을 위해 +1/-1/0 채택.
V값은 [-1, +1] 범위에 학습됨. 0~1 범위가 필요하면 `(V+1)/2`로 변환.

### 4-4. 자가 대국 시 **별도 인스턴스 두 개**

같은 `TDAgent` 인스턴스를 X와 O 양쪽에 쓰면 동일한 `values` dict에 두 시점 값이 섞여 오염됩니다. 정확한 대안은 두 가지:
1. **별도 인스턴스 두 개** (본 프로젝트 채택) — 단순, 명확
2. 상태를 "현재 플레이어 관점"으로 정규화 (코드 복잡도↑)

### 4-5. 게임 종료 후 패자 측 별도 observe

`play_game` 루프 안의 `observe`는 "방금 둔 사람"에게만 호출됩니다. 게임이 끝나면 마지막에 두지 **않은** 쪽(패자 또는 무승부 측)에게도 `-1` 또는 `0` 신호가 들어가야 학습이 됩니다. 이를 위해 루프 종료 후 별도로 `observe(done=True, ...)`를 호출.

이때 `curr_afterstate`로 종료 보드(다른 시점의 보드)를 그대로 넘기는 것은 다소 어색하지만, `TDAgent.observe` 내부의 `if done: values[curr_afterstate] = reward` 로직 덕에 정상 동작합니다 (각 에이전트가 독립된 dict를 가지므로 다른 에이전트와 충돌 없음).

### 4-6. `evaluate`의 training 플래그 백업/복원

평가 중 `agent.training = False`로 강제하면 ε=0이 적용되고 갱신이 차단됩니다. 평가가 끝나면 `try/finally`로 원래 값을 복원해 학습 루프에 부작용을 남기지 않습니다.

### 4-7. QAgent의 `_pending` 상태 관리

Q-learning은 "직전 (s,a)에 대한 갱신을 다음 의사결정 시점의 max로 부트스트랩"하는 구조라, `observe()`만으로는 갱신 시점을 분리하기 어렵습니다. 그래서:
- **비종료 갱신** → 다음 `choose_action` 시작부에서 (s_{t+1}을 보고 max 계산)
- **종료 갱신** → `observe(done=True)`에서 즉시 (target = reward)
- 평가 모드에서는 `_pending` 갱신 자체를 스킵 (학습용 _pending이 평가에 의해 오염되지 않도록)

### 4-8. matplotlib `Agg` 백엔드 강제

`utils.py`에서 import 시점에 `matplotlib.use("Agg")`로 헤드리스 백엔드 강제. SSH/CI 환경에서도 PNG 저장만 잘 되면 OK.

### 4-9. 시드는 실험 스크립트 책임

에이전트 내부에 별도 RNG를 두지 않고, `random` 모듈 전역 시드만 사용. 실험 진입점에서 `set_seed(SEED)` 한 번 호출하면 그 실행의 모든 무작위 분기가 결정론적이 됩니다 (YAGNI).

---

## 5. 유의할 점

### 5-1. ε 설정 vs `learn_from_exploration`은 다른 차원

| | greedy 수 갱신 | 탐험 수 갱신 |
|---|---|---|
| `learn_from_exploration=False` (default) | ✅ | ❌ skip |
| `learn_from_exploration=True` | ✅ | ✅ |

`epsilon`은 **행동 선택**의 무작위성, `learn_from_exploration`은 **학습 갱신** 여부.
Ex 1.4(on/off-policy 비교)의 핵심 변수가 후자입니다.

### 5-2. 평가 시 ε

`evaluate()`는 `training=False`로 강제 → 내부적으로 `effective_epsilon=0`. **평가 시점에 일부러 ε을 살리려면** ex3/ex4의 `evaluate_with_eval_epsilon` 함수 패턴을 사용하세요 (training=True로 두고 ε만 임시로 바꾼 뒤 `play_game(learn=False)`).

### 5-3. 자가 대국 시 같은 인스턴스 금지

`play_game(env, td, td)` ← **버그**. 반드시 두 인스턴스 사용:
```python
td_x, td_o = TDAgent(...), TDAgent(...)
play_game(env, td_x, td_o)
```

### 5-4. `RandomAgent`의 `was_exploratory=False`

탐욕 정책이 정의되지 않으므로 "탐험" 개념이 적용되지 않습니다. 항상 False로 보고하지만 학습이 없는 에이전트라 영향 없음.

### 5-5. ex2의 비대칭 상대는 결정론이 아닌 확률적

처음에 결정론 LeftBiased를 썼더니 두 정책이 모두 100% 승률로 수렴해 차이가 검출되지 않았습니다. **확률적 비대칭(BiasedRandom)**이어야 가설 검증이 가능합니다.

### 5-6. ex5 head-to-head는 진영 비대칭에 주의

두 에이전트 모두 X로만 학습됐다면, head-to-head에서 X 측이 항상 압도합니다. 알고리즘 자체를 비교하려면 **같은 진영**의 결과를 비교하세요 (ex5.py 참고).

### 5-7. `td_agent.py` self-test의 사전 존재 결함

시드 미고정 + 하드코딩된 `legal_actions=[0,1,2,3]` 사용으로 가끔 `simulate_step`이 합법성 검사에서 실패합니다. **본체 로직은 정상**이며 train.py / ex1 / ex3 / ex4 실행으로 검증됩니다.

### 5-8. matplotlib 백엔드

`utils.py`에서 `Agg`를 강제하므로, **인터랙티브 디스플레이가 필요한 코드를 같이 import하면 충돌**할 수 있습니다 (현재 프로젝트 범위에선 문제 없음).

---

## 6. 가설 검증 결과 요약

`notes/ch01/exercises.md`의 가설 15개를 실험으로 확인한 결과 (자세한 수치는 README.md 참고).

| 가설 | 결과 | 한 줄 요약 |
|------|------|----------|
| H1.1a | ✅ | 자가 대국 무승부 비율 0.45 → 0.63 |
| H1.1b | ✅ | 두 정책의 첫 수 분포 L1 거리 = 2.0 (서로 다른 단일 셀에 수렴) |
| H1.1c | △ | 양쪽 vs Random 승률 비슷, 자가 대국 측이 명확히 더 보수적이지 않음 |
| H1.2a | ✅ | 대칭 활용 시 초기 win rate 0.81 → 0.92, 상태수 1205 → 365 |
| H1.2b | ✅ | 대칭 상대(Random)에선 두 정책 최종 성능 동등 |
| H1.2c | ✅ | 비대칭 상대(BiasedRandom)에선 vanilla 0.98 vs symmetric 0.94 |
| H1.3a | ✅ | ε=0 win 0.87, ε=0.1 win 0.99 |
| H1.3b | △ | ε=0.05~0.4 모두 95%+ — 틱택토 단순성 |
| H1.3c | ✅ | 같은 정책: frozen 0.99 vs noisy 0.94 |
| H1.4a | ✅ | 공통 상태 평균 V — off +0.297, on +0.274 |
| H1.4b | ✅ | frozen — off 0.985 vs on 0.977 |
| H1.4c | ❌ | 격차가 noisy에서 오히려 확대 (가설 부정) |
| H1.5a | △ | TD가 초기 약간 빠름 (Q는 ~9× 상태공간) |
| H1.5b | ✅ | 같은 진영 비교 시 Q ≈ TD |
| H1.5c | ✅ | TD 0.97, Q 0.95 |

**총평**: 11/15 지지, 3/15 부분 지지, 1/15 부정. 부정/부분 지지는 모두 틱택토 환경의 단순성에서 비롯된 것으로 추정.

---

## 7. 제작 과정과 이유

전체 작업은 다음 순서로 진행했습니다 (자세한 로그: `notes/ch01/progress_log.md`).

1. **환경 + TD Agent + Random Agent** (이미 완성된 상태로 시작)
2. **train.py** — 모든 학습 루프의 단일 진입점. 자가 대국/패자 처리/평가 격리를 한 번에.
3. **utils.py** — 모든 실험 스크립트가 공유하는 헬퍼. PNG 저장 백엔드 통일, 대칭(D4) 한 곳에 정의.
4. **ex1 → ex2 → ex3 → ex4** — Exercise 순서대로. 각각 가설 검증 결과를 즉시 콘솔 로그로 판정.
5. **q_agent.py** — Q-learning. TDAgent와 인터페이스 일치 + 내부 _pending로 (s,a) 추적.
6. **ex5** — Q vs TD 비교.
7. **README.md / progress_log.md / MANUAL.md** — 최종 문서화.

### 단계별 핵심 결정과 이유

- **train.py를 먼저** 만든 이유: 모든 실험이 동일한 학습 루프를 쓰도록 강제해야, 실험 간 비교가 공정해집니다.
- **utils.py에 대칭(D4)을 둔** 이유: ex2뿐 아니라 일반적인 정책 분석/회귀 검증에도 쓸 수 있으니 한 곳에 모음.
- **observe 시그니처를 통일**한 이유: 학습 루프에서 분기 없이 호출 가능 → 새 에이전트 추가가 쉬움 (예: ex5에서 QAgent를 그대로 끼워 넣을 수 있었음).
- **각 모듈에 self-test**: 모듈을 단독 실행해 빠르게 검증 가능. 회귀 방지.
- **각 실험 스크립트가 가설별 판정 로그를 출력**: 실행 후 그래프를 안 봐도 결과를 즉시 파악 가능.

### 진행 중 발견하고 수정한 이슈

- **ex2 결정론 상대 → 100% 수렴 문제**: 확률적 비대칭으로 교체.
- **ex4 H1.4c 가설 부정**: 가설 자체를 수정하지 않고 결과를 그대로 보고 (CLAUDE.md의 "가설 수정은 사용자 승인 필요" 원칙 준수).
- **ex5 Q-O가 TD-X에게 0% 승률**: 두 에이전트 모두 X로만 학습된 진영 비대칭 효과임을 깨닫고, "같은 진영" 비교로 해석 보정.

---

## 8. 확장 방향

남은 Chapter 1 관련으로 더 해볼 만한 것들:

- **Optimistic initialization**: `default_value=0.5`로 두면 탐험 유도 효과 — ex3에서 추가 실험 가능
- **Decaying α**: 학습률 감소 스케줄 적용 (Ex 1.4의 "α를 적절히 줄여나간다" 부분)
- **Dual-side training**: 모든 에이전트를 X와 O 양 진영으로 학습시켜 ex5 head-to-head 공정성 개선
- **Visualization 강화**: V값을 보드 위치별로 시각화 (학습 정책의 수 선호 패턴 분석)
- **Symmetry-aware Q-learning**: ex2를 Q에도 확장
- **Eligibility traces (12장)**: TD(λ)로 이전 상태들을 동시 갱신

---

## 9. 라이선스 / 출처

- 책: Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd Ed.), MIT Press
- 본 저장소 코드: 학습 목적으로 작성됨. LICENSE 파일 참고.
