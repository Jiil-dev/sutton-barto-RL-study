# Sutton & Barto RL Study

Sutton & Barto의 *Reinforcement Learning: An Introduction* (2nd Ed.)을 공부하면서
**이론 정리 + 코드 구현 + 실험 검증**을 함께 기록하는 저장소.

## 진행 현황

- [x] **Chapter 1: Introduction (틱택토)** — 완료
  - [x] 1.5절 본문 정리
  - [x] Exercise 1.1 ~ 1.5 이론적 답변
  - [x] 환경 + TD Agent 구현
  - [x] Exercise 검증 실험 (1.1 ~ 1.5)
  - [x] Q-learning 확장
- [ ] Chapter 2: Multi-armed Bandits

## Chapter 1 자료

### 노트 (이론)
- [본문 정리: 1.5절 An Extended Example](notes/ch01/1.5_tic_tac_toe.md)
- [Exercise 답변과 검증 가설](notes/ch01/exercises.md)
- [구현 설계 노트](notes/ch01/implementation_notes.md)
- [구현 진행 로그](notes/ch01/progress_log.md)

### 코드
모듈:
- [`code/ch01/env.py`](code/ch01/env.py) — 틱택토 환경 (Gym-style API)
- [`code/ch01/train.py`](code/ch01/train.py) — 학습 루프 (`play_game`, `train`, `evaluate`)
- [`code/ch01/utils.py`](code/ch01/utils.py) — 시각화·저장·대칭(D4) 헬퍼

에이전트:
- [`code/ch01/agents/random_agent.py`](code/ch01/agents/random_agent.py) — 무작위 베이스라인
- [`code/ch01/agents/td_agent.py`](code/ch01/agents/td_agent.py) — TD(0) afterstate 학습
- [`code/ch01/agents/q_agent.py`](code/ch01/agents/q_agent.py) — Q-learning 확장

실험 (Exercise 검증):
- [`experiments/ex1_self_play.py`](code/ch01/experiments/ex1_self_play.py) — Ex 1.1 자가 대국
- [`experiments/ex2_symmetries.py`](code/ch01/experiments/ex2_symmetries.py) — Ex 1.2 대칭성
- [`experiments/ex3_greedy_only.py`](code/ch01/experiments/ex3_greedy_only.py) — Ex 1.3 탐험 vs 탐욕
- [`experiments/ex4_learn_from_exploration.py`](code/ch01/experiments/ex4_learn_from_exploration.py) — Ex 1.4 on/off-policy
- [`experiments/ex5_q_learning.py`](code/ch01/experiments/ex5_q_learning.py) — Ex 1.5 Q-learning

결과 그래프: [`results/ch01/`](results/ch01/) (`ex{1..5}_*.png`)

## Chapter 1 종합 보고서

각 Exercise의 검증 가설(`notes/ch01/exercises.md`)을 실험으로 확인한 요약.
세부 수치/그래프는 [`results/ch01/`](results/ch01/)와 각 실험 스크립트 출력 참고.

| 가설 | 결과 | 핵심 관찰 |
|------|------|----------|
| **H1.1a** 자가 대국 무승부 비율 ↑ | ✅ 지지 | 슬라이딩 윈도우 무승부율: 초반 0.45 → 후반 0.63 |
| **H1.1b** 무작위/자가 학습 정책의 행동 다름 | ✅ 지지 | 첫 수 분포 L1 거리 = 2.0 (완전히 다른 모드 선택) |
| **H1.1c** 자가 대국 정책이 더 보수적 | △ 부분 지지 | 양쪽 vs Random 승률 비슷(97~98%), 자가 측이 무승부 약간 적음. "보수적"이 명확하진 않음 |
| **H1.2a** 대칭성 활용 → 학습 빠름 | ✅ 지지 | 초기 win rate 0.81→0.92, 상태수 1205→365 (3.3×) |
| **H1.2b** 대칭 상대에선 손해 없음 | ✅ 지지 | 최종 vs Random 승률 vanilla 0.95 ≈ sym 0.98 |
| **H1.2c** 비대칭 상대에선 활용 정책 손해 | ✅ 지지 | vs BiasedRandom: vanilla 0.98 vs sym 0.94 |
| **H1.3a** ε-greedy > greedy-only | ✅ 지지 | ε=0 win 0.87, ε=0.1 win 0.99 |
| **H1.3b** ε에 따른 trade-off | △ 약하게 지지 | ε=0.05~0.4 모두 95%+ — 틱택토는 단순해 큰 ε 페널티 작음 |
| **H1.3c** 평가는 ε=0 | ✅ 지지 | 같은 정책: frozen 0.99 vs noisy 0.94 |
| **H1.4a** on-policy V < off-policy V | ✅ 지지 | 공통 상태 평균 V: off +0.297, on +0.274 (Δ −0.023) |
| **H1.4b** off-policy 정책 이김 (frozen) | ✅ 지지 | off 0.985 vs on 0.977 |
| **H1.4c** noisy 평가에서 격차 축소 | ❌ 부정 | 격차 frozen +0.009 → noisy +0.025 (오히려 확대) |
| **H1.5a** Q ≈ TD 학습 속도 | △ 부분 지지 | TD가 초기 약간 빠름 (Q 상태공간이 9× 더 큰 영향) |
| **H1.5b** Q ≈ TD 직접 대결 | ✅ 지지 | X 진영 승률 Q 0.95 ≈ TD 0.99 (둘 다 X 학습이라 동일 진영 비교) |
| **H1.5c** 둘 다 90%+ vs Random | ✅ 지지 | TD 0.97, Q 0.95 |

### 흥미로운 발견 / 메모
- **H1.1b의 L1 = 2.0**: TD는 tie-breaking이 무작위라도 학습이 진행되면 단일 첫 수로 수렴.
  무작위 상대 학습 정책과 자가 대국 학습 정책은 서로 다른 단일 셀에 수렴 → 분포는 둘 다
  one-hot이지만 다른 위치라 L1 거리가 최대값 2.0.
- **H1.4c 부정**: 가설은 "탐험 환경에서 on-policy가 더 robust"였으나 실제론 off-policy의
  본질적 강함이 노이즈 robustness를 압도. 틱택토의 짧은 에피소드 + 결정론 환경이라
  on-policy의 robustness 이점이 잘 드러나지 않는 것으로 보임.
- **H1.5a의 Q 약간 뒤짐**: Q는 (state, action) 키라서 TD의 (afterstate) 키보다 ~9배 많은
  엔트리. 같은 게임 수에선 sample efficiency가 더 낮음. 본질적으론 동치라고 봐야.
- **H1.5b 진영 비대칭**: 둘 다 X로만 학습됐기 때문에 X 진영의 에이전트가 항상 압도.
  공정 비교는 같은 진영(X)일 때의 vs Random 승률만으로 가능.

## 환경 설정

```bash
git clone https://github.com/<your-username>/sutton-barto-rl-study.git
cd sutton-barto-rl-study
python3 -m venv venv
source venv/bin/activate
pip install -r reqiurements.txt
```

## 실행 방법

```bash
# 모듈 단위 self-test
python code/ch01/env.py
python code/ch01/train.py
python code/ch01/utils.py
python code/ch01/agents/random_agent.py
python code/ch01/agents/td_agent.py
python code/ch01/agents/q_agent.py

# 실험 (PNG는 results/ch01/에 저장)
python code/ch01/experiments/ex1_self_play.py
python code/ch01/experiments/ex2_symmetries.py
python code/ch01/experiments/ex3_greedy_only.py
python code/ch01/experiments/ex4_learn_from_exploration.py
python code/ch01/experiments/ex5_q_learning.py
```
