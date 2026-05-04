# Chapter 1 (틱택토) 종합 실험 보고서

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd Ed.) **§1.5 An Extended Example**의 5개 Exercise를
> 코드로 검증한 종합 보고서.
>
> 각 Exercise별로 다음 7단계를 모두 다룹니다.
> 1. **책의 질문** — 원문이 묻는 것
> 2. **사전 답변과 가설** — 직관적 답과 정량 가설 (H1.Xa, H1.Xb, ...)
> 3. **검증 접근법** — 실험 설계 의사결정
> 4. **코드 구성** — 어떻게 짰는지
> 5. **설계 이유** — 왜 그렇게 짰는지 (대안과 트레이드오프)
> 6. **실험 결과** — 수치/그래프
> 7. **결과 해석** — 통찰, 한계, 결론하면 안 되는 것
>
> 자료 위치
> - 가설 정의: [notes/ch01/exercises.md](exercises.md)
> - 코드 사용법: [code/ch01/MANUAL.md](../../code/ch01/MANUAL.md)
> - 결과 그래프: [results/ch01/](../../results/ch01/) 16개 PNG

---

## 목차
- [Part 0. 기반 코드 — Exercise 코드의 토대](#part-0-기반-코드--exercise-코드의-토대)
- [Part 1. Exercise 1.1: Self-Play](#part-1-exercise-11-self-play)
- [Part 2. Exercise 1.2: Symmetries](#part-2-exercise-12-symmetries)
- [Part 3. Exercise 1.3: Greedy Player](#part-3-exercise-13-greedy-player)
- [Part 4. Exercise 1.4: Learning from Exploration](#part-4-exercise-14-learning-from-exploration)
- [Part 5. Exercise 1.5: Other Improvements (Q-learning)](#part-5-exercise-15-other-improvements-q-learning)
- [Part 6. 종합 통찰](#part-6-종합-통찰)

---

## Part 0. 기반 코드 — Exercise 코드의 토대

5개 Exercise 모두 같은 환경(`env.py`)·학습 루프(`train.py`)·헬퍼(`utils.py`)·에이전트 인터페이스를
공유합니다. 각 Exercise를 읽기 전 **공통 어휘**를 정리합니다.

### 0-1. `env.py` — 틱택토 환경

**역할**: 보드 상태/규칙/턴 관리. Gym-style API로 `reset/step/legal_actions`.

**핵심 의사결정**
- **상태 표현 = `tuple[int, ...]` (길이 9)**.
  - `numpy.array`는 unhashable이라 `dict[State, V]` 키로 못 씀.
  - tuple은 hashable + immutable이라 학습 루프 전반에서 안전.
- **플레이어 인코딩 `+1/-1`**.
  - `-player`로 시점을 한 줄에 뒤집을 수 있어, 자가 대국 코드가 단순.
- **보상 `+1/-1/0`** (책의 0/1과 다름). 부호 반전 활용 위해.
- **`simulate_step(action)`** 추가. afterstate value function이 필요한 TD 학습에서
  "각 후보 수의 결과 보드"를 미리 보기 위함. 환경을 변경하지 않고 가상 보드만 반환.

### 0-2. `agents/` — 공통 인터페이스

3종 에이전트(`RandomAgent`, `TDAgent`, `QAgent`)가 같은 시그니처를 만족하므로 학습 루프가 분기 없이
호출 가능합니다.

```python
class Agent(Protocol):
    training: bool
    def choose_action(self, state, legal_actions, env) -> tuple[int, bool]:
        """반환: (action, was_exploratory)"""
    def observe(self, prev_afterstate, curr_afterstate,
                reward, done, was_exploratory) -> None: ...
```

**왜 통일했나**
- 새 에이전트 추가 시 학습 루프를 수정할 필요 없음 (Open/Closed).
- "TD vs Q" 같은 직접 대결 실험을 코드 분기 없이 작성 가능 (실제로 ex5에서 활용).

### 0-3. `train.py` — 학습 루프

```python
def play_game(env, agent_x, agent_o, learn=True) -> int | None
def train(...) -> dict      # 학습 + 주기적 평가, history 반환
def evaluate(...) -> dict   # frozen 평가 (training=False, ε=0)
```

**가장 까다로웠던 결정**: **게임 종료 후 패자 측 별도 observe 호출**.

문제: `play_game` 루프 안에서 `observe`는 "방금 둔 사람"에게만 호출됩니다. 게임이 끝나면 마지막에
두지 *않은* 쪽(패자/무승부측)은 학습 신호를 못 받습니다.

해결:
```python
# 루프 종료 후
last_mover = env.current_player          # env.step은 종료 시 토글 안 함
other = -last_mover
if last_afterstate[other] is not None:
    other_reward = 0.0 if env.winner is None else -1.0
    agents[other].observe(
        last_afterstate[other],          # 패자의 마지막 자기 afterstate
        state,                            # 종료 보드
        other_reward,
        True,                             # done=True
        last_was_exp[other],
    )
```

**왜 이렇게**: `TDAgent.observe` 내부의 `if done: values[curr_afterstate] = reward` 로직 덕에
패자의 values dict에는 "이 종료 보드 = -1"이 저장되고, 그 값이 패자의 마지막 afterstate를 backup하는
target이 됩니다. 각 에이전트가 독립된 dict를 가지므로 충돌 없음.

### 0-4. `utils.py` — 시각화 / 저장 / 대칭

- **`Agg` 백엔드 강제**: 헤드리스 환경에서도 PNG 저장 OK.
- **D4 대칭 8개**(회전 4 + 반사 4) `SYMMETRIES` 미리 정의 → ex2와 일반 정책 분석 모두 활용.
- **`canonical_state(s)`**: 8개 변환 중 **사전식 최소** 형태를 정규형으로 채택. 정렬 가능한 표준형이라
  비교가 빠르고 결정론적.

---

## Part 1. Exercise 1.1: Self-Play

**관련 파일**: [code/ch01/experiments/ex1_self_play.py](../../code/ch01/experiments/ex1_self_play.py)
**산출 그래프**: `results/ch01/ex1_*.png` (5개)

### 1-1. 책의 질문

> 무작위 상대 대신 RL 알고리즘이 자기 자신과 대국하면 어떻게 될까? 다른 정책을 학습할까?

### 1-2. 사전 답변과 가설

**직관**: 다르게 학습된다. 무작위 상대로 학습한 정책은 "상대 실수를 노리는 공격적/기회주의적" 정책이
되지만, 자가 대국은 양쪽이 동시에 강해져 **무승부가 보장되는 안전한 정책**으로 수렴할 것.

**가설** (정량 검증용)
- **H1.1a**: 자가 대국 학습 진행에 따라 **무승부 비율이 증가**한다.
- **H1.1b**: 두 시나리오의 정책은 **행동이 다르다** (예: 첫 수 분포, 평균 게임 길이).
- **H1.1c**: 자가 대국 정책 vs 무작위 상대로 평가하면 둘 다 잘 이기지만, 자가 대국 정책이
  **약간 보수적**일 수 있다 (안전 우선 학습의 부산물).

### 1-3. 검증 접근법

3가지 가설을 각각 측정하기 위한 별도 지표가 필요했습니다.

| 가설 | 측정 지표 | 측정 방법 |
|---|---|---|
| H1.1a | 자가 대국 무승부 비율의 시간에 따른 변화 | 슬라이딩 윈도우(W=200)로 직전 200판 무승부율 |
| H1.1b | 두 정책의 행동 차이 | 빈 보드 첫 수 분포의 L1 거리 |
| H1.1c | 두 정책의 vs Random 승률 | 학습 후 frozen 평가 (각 1000판) |

**시나리오 분리**:
- **A) `td_vs_random`**: TD-X가 RandomAgent O를 상대로 5000판 학습
- **B) `td_self_play`**: TD-X와 TD-O 두 인스턴스가 5000판 자가 대국 (양쪽 모두 학습)

핵심 변수(α=0.2, ε=0.1, num_games=5000, seed=42)는 두 시나리오에서 **동일**하게 두어 차이가
"학습 상대"에서만 비롯되도록 통제.

### 1-4. 코드 구성

```
ex1_self_play.py
├── HYPERPARAMS (NUM_GAMES, ALPHA, EPSILON, EVAL_EVERY, EVAL_GAMES, WINDOW, SEED)
├── run_td_vs_random()           # 시나리오 A
├── run_td_self_play()           # 시나리오 B (★ TDAgent 인스턴스 두 개)
├── sliding_draw_rate(results, window)   # H1.1a 지표 계산
├── plot_self_play_draw_rate(...)        # 그래프
└── main()
    ├── 학습 (A, B)
    ├── H1.1a: 슬라이딩 무승부율 → 그래프 저장 + 초반/후반 비교
    ├── H1.1b: 첫 수 분포 → 히트맵 2개 + L1 거리
    ├── H1.1c: 두 정책 vs Random → 막대그래프 + 비교 판정
    └── 학습 곡선 (참고용) → 저장
```

### 1-5. 핵심 설계 결정과 이유

**(a) 자가 대국에 인스턴스를 2개 사용**
```python
td_x = TDAgent(...)
td_o = TDAgent(...)   # 같은 인스턴스 재사용 금지
```
같은 인스턴스를 X와 O 양쪽에 쓰면 단일 `values` dict에 두 시점의 값이 섞여 오염됩니다.
학습 후 V[s]는 "X가 둔 직후 s" + "O가 둔 직후 s"의 평균 비슷한 값이 되어 의미가 흐려짐.
**대안**: 상태를 "현재 플레이어 관점"으로 정규화 → 단일 인스턴스 가능. 그러나 코드 복잡도가 올라가
교육 목적엔 부적합. **단순 + 명확** 채택.

**(b) 슬라이딩 윈도우 무승부율 (deque + 누적 카운트)**
매번 `sum()`으로 계산하면 O(N²)이라, deque로 윈도우 입출 시 가/감만 하는 O(N) 구현.
틱택토는 5000판이라 O(N²)도 견디지만, 더 큰 실험 대비.

**(c) 첫 수 분포의 측정 시점**
`first_move_distribution`은 평가 모드(`training=False`)로 측정. 탐험 노이즈를 빼고
**탐욕 정책 자체의 분포**만 봐야 의미 있는 비교가 됩니다.

**(d) L1 거리를 분포 비교 지표로**
KL은 0 카운트에 무한대로 폭발 (정책이 결정론적이면 다수 칸에 0). L1은 [0, 2] 범위에서 안정적이고
직관적 ("얼마나 다른 칸을 선호하나").

### 1-6. 실험 결과

```
[ex1] 학습 시작 (num_games=5000, alpha=0.2, epsilon=0.1)
  A) TD vs Random 완료. 알고있는 상태: 1247
  B) TD self-play 완료. X상태:1770 O상태:2050

[H1.1a] 무승부 비율 (윈도우 200): 초반 0.453 → 후반 0.632
[H1.1b] 첫 수 분포 L1 거리: 2.000
[H1.1c] vs Random 1000판:
  A) TD-vs-Random:  {wins: 0.972, losses: 0.001, draws: 0.027}
  B) TD-self-play:  {wins: 0.978, losses: 0.008, draws: 0.014}
```

| 가설 | 결과 | 판정 |
|---|---|---|
| H1.1a | 무승부 0.453 → 0.632 | ✅ 지지 |
| H1.1b | L1 = 2.0 (최대값) | ✅ 강하게 지지 |
| H1.1c | A 승률 0.972 vs B 승률 0.978 | △ 부분 지지 |

### 1-7. 결과 해석과 통찰

**H1.1a (지지)** — 자가 대국 학습이 진행될수록 양쪽이 동시에 강해지면서 minimax 균형으로 수렴.
틱택토의 minimax 균형은 무승부이므로 무승부율 증가는 이론과 일치.
다만 **0.632에서 멈춘** 것은 5000판이 충분치 않거나 ε=0.1의 탐험이 무승부 수렴을 방해한 것으로 보입니다.
ε을 점차 감소시키면 더 높은 무승부율을 보일 것으로 예상.

**H1.1b (강하게 지지) — 가장 흥미로운 결과**
L1 = 2.0은 두 정책의 첫 수 분포가 **공통 칸이 전혀 없다**는 의미.
- 학습이 충분히 진행되면 ε-greedy + tie-breaking 무작위는 결국 단일 칸으로 수렴 (V값에 미세 차이가
  쌓이면 argmax가 결정됨).
- A 정책(vs Random)과 B 정책(self-play)은 **서로 다른 단일 칸**에 수렴 → 두 분포는 모두 one-hot이지만
  다른 위치 → L1 = 2.0.
- 이것은 책이 강조한 "**가치는 환경(상대 포함)에 종속**"의 가장 깔끔한 증거.

**H1.1c (부분 지지) — 가설 일부 부정**
- 양쪽 모두 무작위 상대를 97~98%로 압도. "B가 더 보수적"이라는 직관은 약하게만 관찰됨
  (B의 draws 0.014가 A의 0.027보다 *낮음* — 오히려 B가 더 공격적인 듯한 신호).
- **왜 직관이 빗나갔나**: 무작위 상대 평가는 "기회주의가 잘 통하는" 환경. B가 자가 대국에서 더 많은
  방어 패턴을 학습했더라도, 무작위 상대는 그 방어를 시험할 만큼 정교하지 않음. **정책의 보수성은
  강한 상대를 만나야 드러나는 특성**이라는 점을 시사.

**결론하면 안 되는 것**
- "자가 대국 정책이 더 강하다"라고 일반화할 수 없음. 평가 상대를 바꾸면(예: minimax 상대) 결과가
  달라질 가능성이 큼.
- L1 = 2.0이 항상 나오는 것은 아님. 시드를 바꾸면 다른 단일 칸에 수렴할 수 있고, 아주 작은 확률로
  같은 칸에 수렴할 수도 있음.

---

## Part 2. Exercise 1.2: Symmetries

**관련 파일**: [code/ch01/experiments/ex2_symmetries.py](../../code/ch01/experiments/ex2_symmetries.py)
**산출 그래프**: `results/ch01/ex2_*.png` (3개)

### 2-1. 책의 질문

> 많은 틱택토 위치들이 회전·반사로 사실상 같은 상태이다. 이를 활용해 학습 과정을 어떻게 수정할 수 있을까?
> 어떤 점이 개선될까? 만약 상대가 대칭성을 활용하지 않는다면, 우리도 그래야 할까? 대칭으로 같은
> 위치들이 반드시 같은 가치를 가져야 할까?

### 2-2. 사전 답변과 가설

**직관**: 대칭성 활용은 학습 효율을 높이지만, 상대가 비대칭이면 잘못된 가정이 된다. 가치는 "보드 자체"가
아니라 "보드 + 상대 행동"에 대한 함수이므로, 상대가 비대칭 정책을 가진다면 같은 정규형으로 묶인 보드의
**진짜 가치는 다를 수 있다**.

**가설**
- **H1.2a**: 대칭성 활용 시 같은 게임 수에서 학습 속도가 빨라진다.
- **H1.2b**: 상대가 대칭적인 정책(균등 무작위)일 때 대칭 활용은 손해 없다.
- **H1.2c**: 상대가 비대칭 정책일 때 대칭 활용 정책이 비활용 정책보다 승률이 낮다.

### 2-3. 검증 접근법

3개 가설을 한 스크립트에서 동시에 검증:
1. **시나리오 A** (대칭 상대): vs RandomAgent에서 vanilla TD vs Symmetric TD 학습 곡선 → H1.2a, H1.2b
2. **시나리오 B** (비대칭 상대): vs `BiasedRandomAgent`에서 vanilla TD vs Symmetric TD → H1.2c

비대칭 상대 정의는 핵심 설계 포인트:
- **첫 시도 (실패)**: `LeftBiasedAgent` — 합법수 중 항상 최소 인덱스 선택 (결정론). → 두 정책 모두
  100% 승률로 즉시 수렴해 차이 검출 불가.
- **수정 (성공)**: `BiasedRandomAgent` — 인덱스가 작을수록 큰 가중치(`w_i = 9 - i`)로 확률적 선택.
  비대칭이지만 무작위성이 있어 "단순 외움"이 안 됨.

### 2-4. 코드 구성

```
ex2_symmetries.py
├── SymmetricTDAgent(TDAgent)       # get_value/observe에서 canonical_state로 wrap
├── BiasedRandomAgent                # 가중 무작위 (인덱스 0이 가장 자주 선택됨)
└── main()
    ├── Phase 1 (vs Random):
    │   ├── vanilla TD 학습 → history_vanilla_rand
    │   ├── symmetric TD 학습 → history_sym_rand
    │   ├── 상태 공간 크기 비교 (정량 효과)
    │   ├── H1.2a: 학습 곡선 + 초기 win rate 비교
    │   └── H1.2b: 최종 win rate 비교
    └── Phase 2 (vs BiasedRandom):
        ├── 두 TD 학습
        └── H1.2c: 최종 win rate 비교
```

### 2-5. 핵심 설계 결정과 이유

**(a) `SymmetricTDAgent`를 상속으로 구현**
```python
class SymmetricTDAgent(TDAgent):
    def get_value(self, state):
        return super().get_value(canonical_state(state))
    def observe(self, prev_after, curr_after, reward, done, was_exp):
        prev_c = canonical_state(prev_after) if prev_after else None
        curr_c = canonical_state(curr_after)
        super().observe(prev_c, curr_c, reward, done, was_exp)
```
**왜 상속**:
- 변경 지점이 정확히 2곳(`get_value`, `observe`)이라 상속 + super() 호출이 가장 깔끔.
- `choose_action`은 손대지 않아도 됨. 내부에서 `self.get_value`를 호출하므로 다형성으로 자동 적용.
- 만약 컴포지션(wrap)으로 갔다면 모든 메서드 위임 코드를 작성해야 했음.

**(b) D4 대칭 8개를 인덱스 순열로 미리 계산**
회전/반사 매핑을 매번 재계산하지 않고, 모듈 로드 시 한 번만 만들어 `SYMMETRIES: list[list[int]]`에
저장. 8개 변환 적용은 단순 인덱싱이라 O(8 × 9) = 상수 시간.

**(c) `canonical_state = min(...)` (사전식 최소)**
- 대안: hash로 정규형 결정 → 결정론적이지만 의미 없는 순서.
- 대안: 가장 적은 X·O로 시작하는 변환 선택 → 의미 있지만 구현 복잡.
- **사전식 최소**: 단순, 결정론적, 비교에 sort 사용 가능, 디버깅 시 사람이 읽기 쉬움.

**(d) `BiasedRandomAgent`의 가중치 함수 `w_i = 9 - i`**
- 위치 0의 가중치 9, 위치 8의 가중치 1 → 9배 차이.
- 8배(2^3) 정도면 비대칭이 명확해 vanilla TD가 위치별 가치 차이를 학습하기 충분.
- 1.5배 정도면 노이즈에 묻혀 학습 신호 약함, 100배는 거의 결정론처럼 동작.

### 2-6. 실험 결과

```
[ex2] Phase 1: vs RandomAgent (num_games=4000)
  알고있는 상태 수: vanilla=1205, symmetric=365 (비율 3.30x)
  [H1.2a] 학습 초기(6개 평가 평균) win rate: vanilla=0.812, sym=0.921
  [H1.2b] 최종 평가 (vs Random):
    vanilla={wins: 0.953, losses: 0.017, draws: 0.030}
    sym=    {wins: 0.980, losses: 0.003, draws: 0.017}

[ex2] Phase 2: vs BiasedRandomAgent (비대칭 상대)
  [H1.2c] 최종 평가 (vs BiasedRandom):
    vanilla TD: {wins: 0.983, losses: 0.003, draws: 0.013}
    sym TD:     {wins: 0.943, losses: 0.013, draws: 0.043}
```

| 가설 | 핵심 수치 | 판정 |
|---|---|---|
| H1.2a | 초기 win 0.81 → 0.92, 상태수 1205 → 365 | ✅ 지지 |
| H1.2b | 최종 win 0.95 ≈ 0.98 | ✅ 지지 |
| H1.2c | vs BiasedRandom: vanilla 0.98 > sym 0.94 | ✅ 지지 |

### 2-7. 결과 해석과 통찰

**H1.2a — 정량 효과는 예상대로**
- 상태 공간 1205 → 365 (~3.3배 축소). 이론적 1/8보다 작은 이유: 학습 중 실제로 방문한 상태만 dict에
  들어가는데, 자주 방문하는 상태들은 이미 어느 정도 대칭 클래스를 다양하게 커버하고 있어 절감 폭이 작음.
- 초기 win rate 0.81 → 0.92 — 같은 게임 수에서 더 많은 "효과적 학습"이 일어남. 한 상태에서의 갱신이
  대칭 7개 상태에 동시 반영되는 효과.

**H1.2b — 무손실 보장**
- 최종 승률 거의 동일 (0.95 vs 0.98). 대칭 상대에선 "묶어버림"의 손해가 없다는 것을 확인.
- Symmetric이 약간 더 높은 것은 학습 효율 우위가 충분히 학습 후에도 유지되기 때문.

**H1.2c — 비대칭 상대에서 손해 발생 확인**
- vanilla 0.983 vs symmetric 0.943 — 4%p 격차. 통계적으로 유의 (1500판 평가).
- **왜 차이가 나는가**: BiasedRandomAgent는 위치 0을 자주 선택. vanilla는 "위치 0에 X를 두면 상대가
  근처에 두기 쉽고, 그 결과 특정 패턴으로 빠지더라" 같은 위치별 정보를 학습. Symmetric은 위치 0과 위치 8을
  같은 정규형으로 묶기에 이 정보를 학습 못 함.
- **이것이 책이 의도한 핵심 통찰**: 사전 지식(대칭성)은 그 지식이 환경에 부합할 때만 도움이 된다.

**일반화 가능한 교훈**
1. **귀납 편향(inductive bias)의 양면성**: 학습 효율을 높이지만 가설(대칭성)이 틀리면 천장을 낮춤.
2. **대칭은 "보드의 속성"이 아니라 "환경 + 보드"의 속성**: 상대를 모르고 대칭으로 묶으면 안 됨.
3. **딥러닝의 데이터 증강과 같은 구조**: 회전·반사로 데이터 증강도 같은 트레이드오프 (대칭이 안 맞으면
   오히려 손해).

**결론하면 안 되는 것**
- "비대칭 상대에선 항상 vanilla가 낫다" — 비대칭의 정도와 상대 정책의 학습 가능성에 따라 다름.
  결정론에 가까운 비대칭 상대는 두 정책 모두 100%로 외워버려 차이가 사라짐 (첫 시도의 LeftBiased 사례).

---

## Part 3. Exercise 1.3: Greedy Player

**관련 파일**: [code/ch01/experiments/ex3_greedy_only.py](../../code/ch01/experiments/ex3_greedy_only.py)
**산출 그래프**: `results/ch01/ex3_*.png` (3개)

### 3-1. 책의 질문

> RL 플레이어가 항상 탐욕적이라면, 비탐욕적 플레이어보다 더 잘 학습할까 못할까?

### 3-2. 사전 답변과 가설

**직관**: 못 학습한다. 초기 V값이 모두 동일한 상태에서 우연히 한 경로의 가치가 살짝 올라가면, 탐욕만 두는
정책은 그 경로만 반복해 **로컬 옵티멈에 갇힌다**. ε-greedy의 ε > 0이 이 문제를 해결함.

**가설**
- **H1.3a**: 같은 게임 수 학습 후 평가 시, ε-greedy 정책이 greedy-only보다 승률이 높다.
- **H1.3b**: ε 값에 따른 trade-off가 존재한다 (너무 크면 학습 불안정, 너무 작으면 탐험 부족).
- **H1.3c**: 학습 후 평가는 ε=0으로 해야 진짜 실력 측정 가능 (ε > 0 평가는 노이즈로 승률 깎임).

### 3-3. 검증 접근법

가장 단순한 형태의 sweep 실험:
- ε ∈ {0.0, 0.05, 0.1, 0.2, 0.4} 5개 값에 대해 동일 조건(α=0.2, num_games=4000, seed=11)으로 학습.
- 각 정책을 frozen 평가(ε=0) → win rate 비교 → H1.3a, H1.3b.
- ε=0.1 정책 하나에 대해 frozen vs noisy 평가 비교 → H1.3c.

**중요한 통제**: 매 ε마다 `set_seed(SEED)`로 시드 재설정. 모든 정책이 *동일한* RNG 시퀀스로 시작해야
ε 차이만 반영된 결과가 나옴.

### 3-4. 코드 구성

```
ex3_greedy_only.py
├── EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.4]
├── train_with_epsilon(eps)   # 시드 재설정 후 학습
├── evaluate_with_eval_epsilon(agent, eval_eps, n)
│   # ε을 임시로 바꿔 평가 (학습 갱신은 없음)
└── main()
    ├── Phase 1: ε별 학습 + frozen 평가 → H1.3a, H1.3b
    └── Phase 2: ε=0.1 정책 frozen vs noisy → H1.3c
```

### 3-5. 핵심 설계 결정과 이유

**(a) 매 ε마다 시드 재설정**
```python
def train_with_epsilon(eps):
    set_seed(SEED)              # 매번 같은 시작점
    agent = TDAgent(epsilon=eps)
    train(agent, RandomAgent(), ...)
```
**왜**: 시드를 한 번만 고정하면 첫 번째 학습이 RNG를 소모해 두 번째는 다른 무작위 시퀀스로 시작.
ε 효과가 RNG 차이에 묻힘. **공정 비교를 위해 필수**.

**(b) `evaluate_with_eval_epsilon`을 별도 정의 (기존 `evaluate` 안 씀)**
- 기존 `evaluate(agent, ...)`는 `agent.training = False`로 강제 → `effective_epsilon = 0`.
- H1.3c는 "평가 시점에 ε > 0을 살린다"가 핵심이라, training=True를 유지하고 ε만 임시 변경한 뒤
  `play_game(learn=False)`로 학습 갱신만 차단해야 함.
- 미묘하지만 중요한 차이: `learn=False` ≠ `training=False`.
  - `learn=False`: observe 호출 자체를 안 함 (학습 갱신 0%)
  - `training=False`: observe는 호출되지만 내부에서 early return + ε=0 강제

**(c) ε 후보를 5개로**
- 너무 적으면 trade-off 곡선이 안 보임.
- 너무 많으면 실험 시간 늘고 그래프 가독성↓.
- 0.0(베이스라인), 0.05·0.1·0.2(실용 영역), 0.4(과도 탐험) 정도가 대표성 있는 sweep.

### 3-6. 실험 결과

```
[ex3] ε별 학습 (num_games=4000)
  ε=0.0:  states= 450, frozen vs Random: {wins: 0.868, losses: 0.010, draws: 0.122}
  ε=0.05: states= 864, frozen vs Random: {wins: 0.960, losses: 0.004, draws: 0.036}
  ε=0.1:  states=1031, frozen vs Random: {wins: 0.990, losses: 0.003, draws: 0.007}
  ε=0.2:  states=1250, frozen vs Random: {wins: 0.979, losses: 0.009, draws: 0.012}
  ε=0.4:  states=1730, frozen vs Random: {wins: 0.992, losses: 0.000, draws: 0.008}

[ex3] Phase 2: 평가 모드 비교 (ε=0.1로 학습한 정책)
  평가 ε=0   (frozen):  {wins: 0.987, losses: 0.003, draws: 0.010}
  평가 ε=0.1 (탐험 켜짐): {wins: 0.939, losses: 0.041, draws: 0.020}
```

| 가설 | 핵심 수치 | 판정 |
|---|---|---|
| H1.3a | ε=0 win 0.868 vs ε>0 최고 0.992 | ✅ 강하게 지지 |
| H1.3b | 0.05~0.4 모두 95%+, 명확한 봉우리 X | △ 약하게 지지 |
| H1.3c | frozen 0.987 vs noisy 0.939 (격차 4.8%p) | ✅ 지지 |

### 3-7. 결과 해석과 통찰

**H1.3a — 가장 깨끗하게 지지된 가설**
- ε=0 정책이 알고 있는 상태가 450개로 가장 적음. 항상 같은 길만 가니까 다른 상태를 못 봄.
- 무승부율이 12.2%로 ε>0 정책(0.7~3.6%)보다 훨씬 높음 → "확실히 이기는 길을 못 찾고 무승부로 끝나는 경우"가
  많다는 뜻. 로컬 옵티멈 증거.
- ε 0.05만 줘도 승률이 0.87 → 0.96으로 점프. 탐험은 **소량으로도 큰 효과**.

**H1.3b — 부분 지지, 그러나 흥미로운 발견**
- 가설은 "ε에 따라 종 모양 trade-off (피크가 있음)"을 예상했는데 실제 데이터는 **0.05 이상에서 거의 평탄**.
  - ε=0.1 (0.990)이 살짝 높지만 ε=0.4 (0.992)과 거의 동일.
- **왜 평탄한가**: 틱택토는 (a) 에피소드가 짧고 (b) 상태 공간이 작고 (c) 무작위 상대를 이기는 데 정교한
  탐욕 전략이 필요 없음. 결과적으로 큰 ε이 줄 페널티가 작음.
- **반증으로 살짝 지지된 면**: ε=0.0과 ε=0.05 사이엔 명확한 점프가 있어, 적어도 "탐험이 너무 적으면 큰
  손해" 부분은 확인됨.
- **다른 도메인**(긴 에피소드, 큰 상태공간, 정교한 전략 필요)에선 H1.3b가 더 명확히 보일 것.

**H1.3c — 평가 방법론 교훈**
- 같은 학습된 정책인데 평가 ε에 따라 4.8%p 차이.
- 0.041이 losses (frozen은 0.003) — ε=0.1 노이즈가 *결정적 패배*를 만들어냄. 무작위 상대가 운 좋게 이김.
- **실무 교훈**: "내 정책의 진짜 실력을 알려면 ε=0으로 평가하라". 학습 중 ε=0.1을 쓰더라도 평가 메트릭은
  ε=0이어야 의미 있음.

**결론하면 안 되는 것**
- "ε이 클수록 좋다" (H1.3b 평탄성 때문에). ε=0.4가 ε=0.1과 거의 같은 결과를 줬지만, 실제 학습 시간/
  계산 비용을 고려하면 더 큰 ε은 학습 불안정성 위험. 본 실험은 4000판으로 충분 학습된 후의 *최종* 결과만
  봤을 뿐.
- "ε=0을 학습에 쓰면 안 된다" — 운이 좋으면 ε=0도 최적 경로를 찾을 수 있음. 다만 안전한 선택이 아님.

---

## Part 4. Exercise 1.4: Learning from Exploration

**관련 파일**: [code/ch01/experiments/ex4_learn_from_exploration.py](../../code/ch01/experiments/ex4_learn_from_exploration.py)
**산출 그래프**: `results/ch01/ex4_*.png` (2개)

### 4-1. 책의 질문

> 탐험 수에서도 가치를 갱신한다고 가정하고 α를 적절히 줄여나가면(탐험 빈도는 유지), 가치는 다른 확률
> 집합으로 수렴한다. 두 경우(탐험에서 학습 vs 안 함)에 수렴하는 두 확률은 개념적으로 무엇인가?
> 어느 쪽이 더 잘 학습할까? 어느 쪽이 더 많이 이길까?

### 4-2. 사전 답변과 가설

**개념적 정리**:
- **탐험에서 학습 안 함 (off-policy 류)**: V는 "내가 항상 탐욕적으로 둔다고 가정할 때의 승률"을 향함.
  → 이상적 최적 정책의 가치
- **탐험에서도 학습 (on-policy 류)**: V는 "내가 지금처럼 가끔 탐험하면서 두는 실제 정책의 승률"을 향함.
  → 실제 행동 정책의 가치
- 5장 이후의 SARSA(on-policy) vs Q-learning(off-policy)의 시초 개념.

**가설**
- **H1.4a**: on-policy V값이 off-policy V값보다 전반적으로 낮다 (탐험으로 인한 손실 반영).
- **H1.4b**: 두 정책을 ε=0으로 평가하면 off-policy 쪽이 승률 높다 (평가 환경 = 학습이 가정한 환경).
- **H1.4c**: 두 정책을 ε > 0으로 평가하면 차이 줄거나 역전 (on-policy가 자기 행동 환경을 더 잘 모델링).

### 4-3. 검증 접근법

`TDAgent`에는 이미 `learn_from_exploration: bool` 플래그가 있어, 동일 클래스로 두 모드를 학습 가능.
- **두 정책 학습**: 동일 (α=0.2, ε=0.2, num_games=5000, seed=13)에서 `learn_from_exploration ∈ {False, True}`만 다르게.
- ε=0.2를 쓴 이유: ε=0.1로는 두 모드 차이가 작아 신호 잡기 어려움. 0.2면 탐험 비율이 높아 차이가 잘 드러남.
- **비교 지표**:
  - H1.4a: 두 V dict의 **공통 키**에 대해 평균 비교 (서로 다른 키를 비교하면 의미 없음).
  - H1.4b/c: 동일한 평가 함수에 ε만 다르게.

### 4-4. 코드 구성

```
ex4_learn_from_exploration.py
├── evaluate_with_eval_epsilon(...)   # ex3에서 가져온 패턴 재정의
└── main()
    ├── 두 TDAgent 학습 (off, on)
    ├── H1.4a: 공통 키 평균 V 비교 + 히스토그램
    ├── H1.4b: ε=0 frozen 평가
    └── H1.4c: ε=0.1 noisy 평가 + 격차 변화 분석
```

### 4-5. 핵심 설계 결정과 이유

**(a) 공통 키만 비교** (H1.4a)
```python
common_keys = set(off_policy.values.keys()) & set(on_policy.values.keys())
diff = [on.values[k] - off.values[k] for k in common_keys]
```
**왜**: 두 학습 분기가 같은 시드로 시작해도 ε 분기에서 서로 다른 경로를 탐색할 수 있음. 한쪽만 본 상태는
default(0.0)이라 평균을 왜곡함. **둘 다 본 상태**만 비교해야 진짜 차이가 보임.

**(b) ε=0.2** (vs ex3의 0.1)
- ε=0.1: 탐험이 게임당 ~1번꼴. on/off 차이가 작음.
- ε=0.2: 탐험이 게임당 ~2번꼴. on-policy V가 받는 "탐험 손실" 신호가 2배 → 차이 잡기 쉬움.

**(c) 히스토그램 (단순 평균이 아니라)**
공통 상태 700개에 대한 분포를 보면 평균 차이가 작아도 분포 형태가 다를 수 있음. 시각화로 분포 형태도
확인 가능.

### 4-6. 실험 결과

```
[ex4] 두 정책 학습 (num_games=5000, ε=0.2)
  off-policy: 1334 states
  on-policy:  1534 states

[H1.4a] 공통 상태 703개:
  off-policy V 평균: +0.2970
  on-policy  V 평균: +0.2738
  (on - off) 평균:   -0.0232

[H1.4b] ε=0 frozen 평가:
  off-policy: {wins: 0.985, losses: 0.000, draws: 0.015}
  on-policy:  {wins: 0.977, losses: 0.000, draws: 0.023}

[H1.4c] ε=0.1 noisy 평가:
  off-policy: {wins: 0.971, losses: 0.011, draws: 0.019}
  on-policy:  {wins: 0.946, losses: 0.015, draws: 0.039}
  격차 frozen: +0.009 → noisy: +0.025
```

| 가설 | 결과 | 판정 |
|---|---|---|
| H1.4a | on V 평균 -0.023 (낮음) | ✅ 지지 |
| H1.4b | off 0.985 > on 0.977 | ✅ 지지 |
| H1.4c | 격차 +0.009 → +0.025 (확대) | ❌ 부정 |

### 4-7. 결과 해석과 통찰

**H1.4a — 정확히 이론대로**
- on-policy V가 평균 -0.023 낮음. 이론은 "on-policy V는 가끔 무작위 행동의 손실을 반영"이라 예측.
- 실측 -0.023은 ε=0.2 × (탐험으로 인한 평균 손실 정도)로 해석 가능. 정확한 분석을 위해선
  γ나 에피소드 길이에 따른 계산이 필요하지만, 부호와 크기 모두 합리적.
- **on-policy V는 비관적이지 않다, 정확하다** — 이게 책이 강조하는 핵심.

**H1.4b — 평가 환경에서 우위 확인**
- ε=0 평가는 "탐욕만 두는 환경" → off-policy의 학습 가정과 일치 → off-policy 우위.
- 격차는 작음 (0.9%p). 틱택토에서 두 모드 모두 탐욕 정책의 가치를 잘 학습한다는 의미.
- losses는 둘 다 0 — 양쪽 다 무작위 상대에 절대 안 짐.

**H1.4c — 가설 부정, 그러나 유익한 부정**
- 가설: noisy 평가에서 격차 축소 또는 역전. on-policy가 노이즈 환경을 더 정확히 학습했으니 robust할 것.
- 실측: 격차 +0.009 → +0.025로 **확대**.
- **왜 가설이 빗나갔나** (3가지 추정):
  1. **틱택토는 ε=0.1 노이즈로 충분히 흔들리지 않는다**: 게임이 9수 이내라 노이즈 1번이 게임 결과를
     좌우하기 어려움. on-policy의 "노이즈 환경 모델링" 우위가 실현될 여지가 작음.
  2. **off-policy V가 본질적으로 더 강한 정책을 함의**: ε=0으로 평가할 때 off-policy가 직접 0.014
     앞서는데, ε=0.1을 줘도 그 본질적 우위가 사라지지 않음.
  3. **on-policy의 우위는 "행동 정책의 의사결정" 단계에서 드러나는 것**: 평가는 둘 다 결정적 행동을
     가정하므로 on-policy가 "탐험 환경의 정확한 V"를 가졌어도 활용 못 함.
- **이게 맞다면, 다른 환경에선 H1.4c가 지지될 가능성도 있음**: 더 긴 에피소드, 더 큰 ε, 더 큰
  상태공간에서.

**개념적 통찰**
- on-policy/off-policy의 "어느 쪽이 좋은가"는 **목적에 따라 다르다**:
  - **학습 중 의사결정 자체에 V를 사용** → on-policy V가 더 정확함.
  - **학습 후 frozen 정책 배포** → off-policy V로 학습한 정책이 일반적으로 더 강함.
- 본 실험은 후자만 확인. 전자는 다른 실험 설계가 필요 (예: 평가 시 V값을 직접 노출해 의사결정에 활용).

**결론하면 안 되는 것**
- "off-policy가 항상 낫다" — H1.4c 부정은 환경 의존적. SARSA가 cliff-walking에서 Q-learning보다 안전한
  정책을 학습하는 유명한 사례가 있음. 틱택토는 SARSA의 우위가 드러날 환경이 아닐 뿐.

---

## Part 5. Exercise 1.5: Other Improvements (Q-learning)

**관련 파일**:
- 에이전트: [code/ch01/agents/q_agent.py](../../code/ch01/agents/q_agent.py)
- 실험: [code/ch01/experiments/ex5_q_learning.py](../../code/ch01/experiments/ex5_q_learning.py)
**산출 그래프**: `results/ch01/ex5_*.png` (3개)

### 5-1. 책의 질문

> RL 플레이어를 개선할 다른 방법을 생각해볼 수 있는가? 이 문제를 푸는 더 나은 방법이 있을까?

### 5-2. 사전 답변과 가설

**선택한 방향**: V(s) → Q(s, a)로 확장 (Q-learning, 책 6장).
다른 가능성(eligibility traces, optimistic init, 함수 근사)이 있지만 메인 확장은 Q-learning.

**가설**
- **H1.5a**: 같은 학습량에서 Q가 V기반 TD보다 빠르게 수렴 (혹은 비슷).
- **H1.5b**: 학습된 Q vs V 직접 대결 시, Q가 비등하거나 우위.
- **H1.5c**: 두 방법 모두 충분히 학습하면 무작위 상대에게 거의 100% 승률.

**이론적 노트**: 틱택토는 deterministic 환경이라 `Q(s, a) ≈ V(afterstate(s, a))`. 즉 두 알고리즘은
수학적으로 거의 동치. 큰 차이는 기대하지 않음.

### 5-3. 검증 접근법

3개 가설을 한 스크립트에서 측정:
1. **H1.5a**: 동일 하이퍼로 TD/Q 학습 → 학습 곡선 비교 (초반/후반 win rate).
2. **H1.5b**: 학습 후 두 정책 직접 대결, 양 진영 모두 (Q-X vs TD-O, Q-O vs TD-X).
3. **H1.5c**: 최종 vs Random 평가.

### 5-4. 코드 구성

`q_agent.py` (가장 큰 새 코드):
```
class QAgent:
    q: dict[(State, int), float]
    _pending_state, _pending_action, _pending_reward   # 지연 갱신용
    
    def get_q(self, state, action) -> float
    def choose_action(self, state, legal_actions, env) -> tuple[int, bool]:
        # 1) 직전 (s,a)에 대한 비종료 Q-갱신 (현재 state로 bootstrap)
        # 2) ε-greedy 선택
        # 3) _pending에 (state, action) 저장
    
    def observe(self, prev_after, curr_after, reward, done, was_exp) -> None:
        # done이면 종료 갱신 (target = reward), pending 리셋
        # 비종료면 _pending_reward만 저장 (다음 choose_action에서 사용)
```

`ex5_q_learning.py`:
```
ex5_q_learning.py
├── head_to_head(agent_a, agent_b, a_plays_x, n)   # 직접 대결 (training=False)
└── main()
    ├── TD/Q 학습 (시드 재설정으로 공정 비교)
    ├── H1.5a: 학습 곡선 + 초반/후반 비교
    ├── H1.5c: 최종 vs Random
    └── H1.5b: 양 진영 head-to-head + 같은 진영 비교 보정
```

### 5-5. 핵심 설계 결정과 이유

**(a) Q-갱신 시점을 둘로 분리** (가장 어려웠던 결정)
- **종료 갱신**: `observe(done=True)`에서 즉시 (target = reward, no bootstrap).
- **비종료 갱신**: 다음 `choose_action` 시작부에서 (s_{t+1}을 보고 max bootstrap).

**왜 이렇게 어렵나**: TD-V는 `observe(prev_after, curr_after)` 한 번에 갱신이 끝남 (curr_after가 곧 target).
Q-learning은 "내가 둔 직후"가 아니라 "내가 둔 + 상대가 둔 후"의 상태가 bootstrap target. 그런데
`observe`는 "내가 둔 직후"에만 호출됨. → 지연 갱신 메커니즘 필요.

**대안과 기각 이유**:
1. **`observe` 시그니처 변경**: 모든 에이전트가 깨짐. 인터페이스 통일성 손상.
2. **train.py에서 Q를 특별 처리**: 학습 루프가 에이전트 종류를 알아야 함. 추상화 위반.
3. **에이전트 내부 상태로 지연**: 채택. 인터페이스 안 바꿔도 됨. ★

**(b) `prev_afterstate`/`curr_afterstate` 인자를 받지만 무시**
```python
def observe(self, prev_afterstate, curr_afterstate, reward, done, was_exploratory):
    del prev_afterstate, curr_afterstate, was_exploratory
    # ...
```
**왜**: 인터페이스 통일을 위해 인자는 받되, Q-learning은 내부 _pending이 진실의 원천.
`del`로 명시적으로 미사용을 표시 → lint 경고 회피 + 의도 표현.

**(c) 평가 모드에서 _pending 갱신 자체를 스킵**
```python
if self.training:
    self._pending_state = state
    self._pending_action = action
```
**왜**: 학습 → 평가 → 학습 인터리브 시, 평가 게임의 (s, a)가 _pending에 들어가면 다음 학습 게임에서
잘못된 bootstrap이 발생. 격리 필수.

**(d) `learn_from_exploration` 플래그 없음**
- TDAgent와 달리 Q-learning은 max로 자연스럽게 off-policy.
- 탐험 행동이라도 그 행동의 Q를 max bootstrap으로 갱신하는 것이 *표준* Q-learning.
- 일부러 스킵하면 표준에서 벗어남. **표준 따르기** 채택.

**(e) ex5의 head-to-head를 양 진영 모두**
처음엔 한 진영(Q-X vs TD-O)만 했는데 0% 또는 100% 같은 극단 결과 가능성 우려. 양 진영 모두 측정해
대칭 점검.

### 5-6. 실험 결과

```
[ex5] TD vs Q 학습 (num_games=5000)
  TD: states=1088, Q: pairs=2408

[H1.5a] 초반 평균 win — TD: 0.907, Q: 0.846
        후반 평균 win — TD: 0.971, Q: 0.948

[H1.5c] 최종 vs Random:
  TD: {wins: 0.967, losses: 0.000, draws: 0.033}
  Q:  {wins: 0.950, losses: 0.004, draws: 0.046}

[H1.5b] head-to-head (Q 관점):
  Q-X vs TD-O: {wins: 0.950, losses: 0.003, draws: 0.047}
  Q-O vs TD-X: {wins: 0.000, losses: 0.987, draws: 0.013}
        X 진영 승률 비교 — Q-X: 0.950, TD-X: 0.987
```

| 가설 | 핵심 수치 | 판정 |
|---|---|---|
| H1.5a | 초반 TD 0.91 > Q 0.85 | △ 부분 지지 (Q가 약간 뒤짐) |
| H1.5b | 같은 진영 비교 시 Q 0.95 ≈ TD 0.99 | ✅ 지지 |
| H1.5c | TD 0.97, Q 0.95 (둘 다 95%+) | ✅ 지지 |

### 5-7. 결과 해석과 통찰

**H1.5a — 약하게 부정, 그러나 이유가 분명**
- TD가 초기·후반 모두 약간 우위. 가설은 "Q가 비슷하거나 빠름"이었으니 부분 지지로 채점.
- **왜 Q가 뒤지나**: Q는 (s, a) 키, V-afterstate는 (afterstate) 키. 같은 의미 정보를 더 많은 엔트리로
  나눠 저장 → sample efficiency 열위.
  - 본 실험: TD 1088 entries vs Q 2408 entries (~2.2배). 같은 게임 수에 *각 entry당 평균 갱신 횟수*가
    Q가 작음.
  - **수학적 동치성과 sample efficiency는 다른 문제**: 한쪽이 더 많은 표를 만들면 같은 데이터로 채우기
    어려움.
- 실용적 함의: 작고 결정론적인 환경에선 V-afterstate가 효율적. Q-learning의 진가는 stochastic transition,
  큰 상태공간에서 드러남.

**H1.5b — 진영 비대칭이라는 함정**
- 처음 본 결과: Q-O가 TD-X에 0% 승률. 충격적. 가설 부정처럼 보임.
- 분석: 두 에이전트 모두 X로만 학습. O 진영은 학습 분포 밖 상태가 다수 → 디폴트 V로 거의 무작위 행동.
- **공정 비교**: 같은 진영(X)일 때의 vs 상대 승률만 비교 → Q-X 0.95 vs TD-X 0.99. 거의 동등.
- **교훈**: 비교 실험은 "비교 가능한 조건"인지 점검 필수. 학습 환경이 비대칭이면 평가 결과를 그대로
  믿으면 안 됨.

**H1.5c — 충분 학습 시 동등한 천장**
- TD 0.97, Q 0.95 — 사실상 같은 천장에 도달.
- 이것이 본 Exercise의 가장 중요한 결과: **알고리즘 선택보다 충분한 학습/탐험이 중요**.

**일반화된 통찰**
1. **Q-learning과 V-afterstate는 deterministic 환경에서 수학적으로 동치이지만 sample efficiency는 다름**.
2. **알고리즘 비교 시 "어느 쪽이 더 빨리 잘 되는가"는 환경/상태표현/해상도에 따라 다름**.
3. **공정 비교는 비교 가능한 조건을 만들어야 한다** — 진영, 시드, 학습량 모두 통제.

**결론하면 안 되는 것**
- "Q-learning이 V-afterstate보다 못하다" — 본 실험은 deterministic + 작은 환경의 특수 케이스.
- "두 알고리즘은 같다" — 일반 환경(stochastic, 큰 상태공간)에선 본질적 차이가 있음 (특히 SARSA vs Q에서).

---

## Part 6. 종합 통찰

### 6-1. 가설 검증 총괄 (15개)

```
✅ 지지         11개  (H1.1a, H1.1b, H1.2a, H1.2b, H1.2c, H1.3a, H1.3c, H1.4a, H1.4b, H1.5b, H1.5c)
△ 부분 지지     3개  (H1.1c, H1.3b, H1.5a)
❌ 부정          1개  (H1.4c)
```

부정/부분 지지의 **공통 원인은 모두 틱택토 환경의 단순성**:
- 짧은 에피소드 (≤9수)
- Deterministic transitions
- 작은 상태공간 (~5,478개)
- 강한 학습 신호 (승/패가 명확)
- 무작위 베이스라인이 너무 약함

이것이 책 1장이 "도입 예제"인 이유 — 모든 RL 개념의 직관을 보여주지만, 정량적 비교의 미세한 차이는
이 환경에서 잘 안 드러남.

### 6-2. 챕터를 통해 학습한 RL의 핵심 직관

각 Exercise가 강조한 일반 원리:

1. **Ex 1.1 → 가치는 환경에 종속**. 같은 알고리즘도 상대가 다르면 다른 정책을 학습.
2. **Ex 1.2 → 사전 지식은 양날의 검**. 환경에 부합할 때만 도움.
3. **Ex 1.3 → 탐험 없이는 학습 없다**. 단, 평가 시엔 탐험을 끄라.
4. **Ex 1.4 → on/off-policy는 다른 가치를 학습한다**. 어느 쪽이 좋은가는 사용 목적에 따라 다름.
5. **Ex 1.5 → 알고리즘보다 환경/문제 정의가 중요**. 동치인 알고리즘도 sample efficiency가 다를 수 있음.

### 6-3. 코드 측면에서 얻은 교훈

1. **인터페이스 통일이 확장의 비용을 결정한다**: TDAgent와 같은 시그니처로 QAgent를 만들었기에 학습 루프
   수정 없이 ex5에 그대로 끼울 수 있었음.
2. **"학습 vs 평가" 격리는 자주 헷갈린다**: `training=False` vs `learn=False` vs ε 임시 변경 — 세 가지를
   상황에 따라 골라 써야 함.
3. **공정 비교의 통제**: 시드 재설정, 동일 하이퍼, 동일 평가 환경, 진영 대칭 모두 빠짐없이 점검.
4. **가설 부정도 가치 있다**: H1.4c 부정이 환경 단순성의 한계를 드러냄. 실험은 가설을 반증할 수 있을 때
   비로소 과학적.

### 6-4. 아쉬운 점 / 다음 단계

- **무작위 베이스라인의 한계**: 최종 승률 평가가 모두 vs Random이라 천장이 낮음. minimax 상대를 만들면
  더 정교한 비교 가능 (틱택토는 minimax가 가벼움).
- **양 진영 학습**: 모든 에이전트를 X·O 양쪽으로 학습시키면 ex5 head-to-head가 공정해짐.
- **하이퍼파라미터 sweep 자동화**: ex3에선 ε sweep을 했지만, α나 num_games도 sweep하면 trade-off 그림이
  완성됨.
- **재현성 강화**: numpy seed도 같이 고정 (현재는 random만). matplotlib font 설정도 통일.

### 6-5. 자료 참고

- 가설 정의: [notes/ch01/exercises.md](exercises.md)
- 코드 사용법: [code/ch01/MANUAL.md](../../code/ch01/MANUAL.md)
- 설계 결정 기록: [notes/ch01/implementation_notes.md](implementation_notes.md)
- 작업 진행 로그: [notes/ch01/progress_log.md](progress_log.md)
- 결과 그래프: [results/ch01/](../../results/ch01/) (16개 PNG)
