# Project Context for Claude Code

## 프로젝트 목적
Sutton & Barto의 *Reinforcement Learning: An Introduction* (2nd Ed.)을 공부하면서
**이론 정리 + 코드 구현 + 실험 검증**을 함께 기록하는 학습 저장소.

저자(사용자)는 RL 입문자이며 인공지능 코딩 경험이 처음. 따라서 코드는
"학습 노트로서도 읽힐 수 있는" 수준의 명확성을 가져야 함.

## 현재 진행 중: Chapter 1 (틱택토)

### 이미 완성된 것
- `notes/ch01/1.5_tic_tac_toe.md` — 본문 번역 + 보충 설명
- `notes/ch01/exercises.md` — Ex 1.1~1.5 이론적 답변과 검증 가설
- `notes/ch01/implementation_notes.md` — 설계 결정 기록
- `code/ch01/env.py` — 틱택토 환경 (Gym-style API + simulate_step)
- `code/ch01/agents/td_agent.py` — TD(0) afterstate 학습 에이전트

### 남은 작업 (이 순서로 진행)
1. `code/ch01/agents/random_agent.py` — 무작위 상대
2. `code/ch01/train.py` — 학습 루프 (env + agent 연결)
3. `code/ch01/utils.py` — 시각화, 저장 헬퍼
4. `code/ch01/experiments/ex1_self_play.py`
5. `code/ch01/experiments/ex2_symmetries.py`
6. `code/ch01/experiments/ex3_greedy_only.py`
7. `code/ch01/experiments/ex4_learn_from_exploration.py`
8. `code/ch01/agents/q_agent.py` — Q-learning 확장
9. `code/ch01/experiments/ex5_q_learning.py`
10. README.md 최종 정리

각 Exercise 스크립트는 `notes/ch01/exercises.md`에 정의된 **검증 가설(H1.xa, H1.xb, ...)** 을
실험으로 확인해야 함.

## 코드 컨벤션

### 일반
- Python 3.10+
- Type hint 적극 사용 (`list[int]`, `int | None` 등 PEP 604 신문법)
- Google-style docstring (Args, Returns, Raises 섹션)
- `if __name__ == "__main__":` 블록에 모듈 단위 self-test 포함
- 매직 넘버 금지 — 의미 있는 상수 정의

### 네이밍
- 환경: `EMPTY=0`, `PLAYER_X=+1`, `PLAYER_O=-1`
- 상태 타입 별칭: `State = tuple[int, ...]`
- 내부 메서드는 `_` 접두사

### 환경 인터페이스 (Gym-style)
- `reset() -> State`
- `step(action) -> tuple[State, float, bool, dict]`  # (next_state, reward, done, info)
- `legal_actions() -> list[int]`
- `simulate_step(action) -> State`  # afterstate 미리보기, 실제 변경 없음
- `done`, `winner`, `current_player`는 `@property`

### 에이전트 인터페이스
모든 에이전트는 다음 메서드를 가져야 함 (TDAgent 참고):
- `choose_action(state, legal_actions, env) -> tuple[int, bool]`  # (action, was_exploratory)
- `observe(prev_afterstate, curr_afterstate, reward, done, was_exploratory) -> None`
- `training: bool` 토글 (학습/평가 모드)

학습 루프는 어떤 에이전트든 동일한 인터페이스로 호출 가능해야 함.

### 실험 스크립트
- 각 스크립트는 한 Exercise의 가설 검증
- 결과 그래프는 `results/ch01/`에 PNG로 저장
- 스크립트 상단 docstring에 "검증할 가설"과 "실험 설계" 명시
- random.seed 고정으로 재현성 보장

### 보상 규약
- 승리: +1, 패배: -1, 무승부: 0, 진행 중: 0
- (책 §1.5의 0/1과 다름. 부호 반전 활용 위해 +1/-1/0 채택)
- V값은 [-1, +1] 범위

## 작업 흐름 요청

남은 작업 목록(1~10번)을 처음부터 끝까지 자율적으로 진행한다.
중간에 승인을 묻지 말고, 다음 원칙에 따라 끝까지 완주할 것:

- 각 모듈 작성 직전, 그 모듈의 설계 결정과 핵심 트레이드오프를
  3~5줄로 요약하고 바로 코드 작성에 들어간다.
- 각 파일 작성 후 자체 self-test(`if __name__ == "__main__":` 블록)를
  반드시 실행해 통과를 확인하고, 실패하면 스스로 수정한다.
- 모듈 단위로 진행 로그를 `notes/ch01/progress_log.md`에 누적 기록
  (만든 파일, 핵심 결정, 테스트 결과, 발견된 이슈).
- 다음 경우에만 멈추고 사용자에게 묻는다:
  (a) 책의 해석이 갈리는 이론적 결정 (e.g., 보상 규약 변경, 
      알고리즘 선택의 본질적 차이)
  (b) `notes/ch01/exercises.md`의 가설 자체를 수정해야 할 때
  (c) 외부 의존성을 추가해야 할 때
- 그 외 코드 스타일/구현 디테일은 자율 판단.
- 모든 작업 완료 후, 최종 종합 보고서 1개만 작성.

## 참고 자료
- 책: Sutton & Barto, RL: An Introduction (2nd Ed.), Chapter 1
- 노트의 검증 가설은 `notes/ch01/exercises.md` 참고
- 설계 결정 추가/변경 시 `notes/ch01/implementation_notes.md`에 기록