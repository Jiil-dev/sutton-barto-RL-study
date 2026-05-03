## 환경 (Environment) 설계

### 상태 표현
**선택: 길이 9의 `tuple[int, ...]`** (예: `(0, 1, -1, 0, ...)`)

후보 비교:
- `numpy.array`: 빠르지만 hashable하지 않아 dict 키로 못 씀
- `tuple`: hashable, immutable, 작은 크기에서 충분히 빠름 ← 선택
- `str`: 직관적이지만 문자/숫자 변환 오버헤드

핵심 이유: **가치 함수가 `dict[State, float]` 구조**이므로 hashable 필수.
또한 immutable이라 실수로 변경될 위험 없음.

### 플레이어 표현
- `EMPTY = 0`, `PLAYER_X = +1`, `PLAYER_O = -1`
- +1/-1로 한 이유: **부호 반전(`-player`)으로 시점 전환** 가능
  → 자가 대국(Ex 1.1) 구현 시 한 에이전트의 두 시점을 통일된 코드로 처리 가능

### 합법 수 판정
`legal_actions()` → 빈 칸 인덱스 리스트.
list comprehension으로 한 줄 구현, O(9) = O(1).

### 승패/무승부 판정
- 8개 승리 라인을 미리 상수로 정의
- `_check_win(player)`: 8라인 중 하나라도 `player`로 가득 차면 승
- `_is_board_full()`: 모든 칸이 EMPTY가 아니면 보드 가득

### 보상 설계
- 승리: +1
- 패배: -1
- 무승부: 0
- 진행 중: 0

책 §1.5는 0/1/0이지만, +1/-1/0이 자가 대국 시 부호 반전 활용에 자연스러움.
가치값 V ∈ [-1, +1] 범위로 학습됨. 0~1 범위 비교가 필요하면 `(V+1)/2`로 변환.

### Gym-style API
`reset()`, `step(action)` → `(next_state, reward, done, info)` 형식.
업계 표준이라 나중에 다른 RL 라이브러리와 호환 용이.

### 캡슐화 원칙
- 내부 보드: 가변 `list` (수정 필요)
- 외부 노출: 불변 `tuple` (안전성)
- `_` 접두사로 내부 멤버 표시
- `@property`로 읽기 전용 속성 노출

---

## 에이전트 (Agent) 인터페이스 규약

### 공통 시그니처
모든 에이전트(`TDAgent`, `RandomAgent`, 추후 `QAgent` 등)는 학습 루프에서
**분기 없이** 호출 가능하도록 동일한 시그니처를 갖는다.

- `choose_action(state, legal_actions, env) -> tuple[int, bool]`
  - 반환: `(action, was_exploratory)`
- `observe(prev_afterstate, curr_afterstate, reward, done, was_exploratory) -> None`
- `training: bool` 어트리뷰트 (학습/평가 모드 토글)

이 통일된 인터페이스 덕분에 `train.py`는 에이전트 종류를 신경 쓰지 않고
같은 루프로 `Random vs TD`, `TD vs TD`(자가대국), `Q vs TD` 등을 돌릴 수 있다.

### `was_exploratory` 의미와 에이전트별 규약
- **TDAgent**: ε-greedy에서 ε 분기로 탐험을 선택했으면 `True`, 탐욕이면 `False`.
  자기 자신의 `observe()`에서 갱신 여부 결정에 쓰임 (기본 설정상 탐험 수에선 갱신 스킵).
- **RandomAgent**: **항상 `False`**.
  근거: "탐험"은 "탐욕 정책에서 의도적으로 벗어났다"는 의미인데 무작위 정책은
  탐욕 정책 자체가 정의되지 않으므로 개념이 적용되지 않음. False가 가장 안전한
  기본값이며, RandomAgent는 학습이 없어 이 플래그의 값이 무엇이든 동작에 영향 없음.

### 학습 없는 에이전트의 인터페이스 처리
- `observe()`는 **no-op**으로 구현 (받은 인자를 모두 무시).
- `training` 토글은 동작에 영향 없는 placeholder로 보유 — 인터페이스 통일이 유일 목적.
- `choose_action`의 `state`, `env` 인자는 미사용이면 `del state, env`로 의도 명시
  (lint 경고 회피 + "사용하지 않음"이 의도임을 코드로 표현).

### 재현성(시드) 처리 책임 분리
- 에이전트 내부에 별도 RNG/seed를 두지 **않는다**.
- `random` 모듈 레벨 RNG를 사용하며, **시드 고정은 실험 스크립트의 책임**.
  (예: `experiments/ex1_self_play.py` 진입점에서 `random.seed(seed)` 호출)
- 이렇게 하면 한 번의 시드 고정으로 모든 에이전트의 무작위 분기가 결정론적이 됨.
- 만약 추후 "두 에이전트가 같은 실행 안에서 독립적인 시드로 재현되어야 한다"는
  요구가 생기면 그때 per-agent `random.Random` 인스턴스를 도입하기로 미룬다 (YAGNI).