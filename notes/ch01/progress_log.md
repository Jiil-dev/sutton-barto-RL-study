# Chapter 1 Progress Log

> CLAUDE.md의 "남은 작업 1~10번"을 자율 실행한 작업 기록.
> 모듈 단위로: 만든 파일 / 핵심 결정 / self-test 결과 / 발견 이슈.

---

## Task 1: `agents/random_agent.py`
- 무작위 정책. `was_exploratory=False` 고정 (탐욕 정책 정의 안 됨 → "탐험" 비적용).
- self-test: 1000회 샘플 합법성 + 분포 균등 + RandomAgent vs RandomAgent 완주 — 통과.

## Task 2: `train.py`
- 핵심 함수: `play_game`, `train`, `evaluate`. `Agent` Protocol로 인터페이스 문서화.
- **결정**: 게임 종료 후 마지막에 두지 않은 쪽(패자/무승부측)에게 별도로 terminal observe 호출.
  curr_afterstate에 종료 보드를 그대로 넘겨, observe() 내부 `if done: values[curr] = reward`
  로직으로 패배(-1)/무승부(0) 신호가 들어가게 함.
- **결정**: 자가 대국은 같은 인스턴스 두 번이 아니라 **별도 인스턴스 두 개** 사용. 이유: 같은
  values dict에 X 시점·O 시점 값이 섞여 오염되기 때문.
- **결정**: `evaluate`는 양 에이전트의 `training` 플래그를 잠시 False로 강제하고 finally에서
  복원. ε=0 + 학습 갱신 없음 보장.
- self-test 7개 모두 통과 (TD 2000판 학습 후 vs Random 88% 승률 확인).

## Task 3: `utils.py`
- 시각화 (Agg 백엔드 강제), pickle 저장/로드, 시드 고정, 첫 수 분포, D4 대칭 8개.
- **결정**: 대칭은 `SYMMETRIES: list[list[int]]` 인덱스 순열. `canonical_state`는 8개 변환 중
  사전식 최소를 정규형으로 채택.
- self-test 7개 모두 통과 (대칭 클래스 크기 검증: 모서리 4, 변 4, 중앙 1).

## Task 4: `experiments/ex1_self_play.py`
- 시나리오 A: TD-X vs Random / 시나리오 B: TD-X vs TD-O 자가 대국.
- 결과 (5000판, α=0.2, ε=0.1, seed=42):
  - **H1.1a 지지**: 자가 대국 무승부 비율 0.45 → 0.63 (윈도우 200)
  - **H1.1b 지지**: 첫 수 분포 L1 거리 2.0 (서로 다른 단일 셀에 수렴)
  - **H1.1c 부분 지지**: 양쪽 vs Random 승률 비슷 (~0.97), "보수적" 차이 미세

## Task 5: `experiments/ex2_symmetries.py`
- `SymmetricTDAgent` (TDAgent 상속, canonical_state로 wrap), `BiasedRandomAgent` (가중 무작위).
- **이슈 발견**: 처음 비대칭 상대로 `LeftBiasedAgent`(결정론)를 썼더니 두 정책 모두 100%
  → 가설 차이 검출 불가. 확률적 비대칭 (`BiasedRandomAgent`)으로 교체.
- 결과 (4000판):
  - 상태 공간: vanilla 1205 → symmetric 365 (~3.3×)
  - **H1.2a 지지**: 초기 win rate 0.81 → 0.92
  - **H1.2b 지지**: 최종 vs Random 거의 동등 (0.95 vs 0.98)
  - **H1.2c 지지**: vs BiasedRandom — vanilla 0.98 vs symmetric 0.94

## Task 6: `experiments/ex3_greedy_only.py`
- ε ∈ {0, 0.05, 0.1, 0.2, 0.4} × 4000판 학습.
- 결과:
  - **H1.3a 지지**: ε=0 win 0.87, ε=0.1 win 0.99
  - **H1.3b 약하게 지지**: 0.05 이상에선 모두 95%+ (틱택토 단순성)
  - **H1.3c 지지**: ε=0.1 정책: frozen eval 0.99 vs noisy eval 0.94

## Task 7: `experiments/ex4_learn_from_exploration.py`
- TDAgent 두 개: `learn_from_exploration={False, True}`, ε=0.2.
- 결과 (5000판):
  - **H1.4a 지지**: 공통 상태 평균 V — off +0.297, on +0.274 (Δ −0.023)
  - **H1.4b 지지**: frozen eval — off 0.985 vs on 0.977
  - **H1.4c ❌ 부정**: 격차가 frozen +0.009 → noisy +0.025로 오히려 확대.
    가설(noisy에서 격차 축소)이 부정됨. 해석: 틱택토 짧은 에피소드 + ε=0.1 작은 노이즈에서
    on-policy의 robustness 이점이 off-policy의 본질적 강함을 압도하지 못함.

## Task 8: `agents/q_agent.py`
- 표준 Q-learning. γ=1, 종료 시 bootstrap 없이 r만.
- **결정**: TDAgent와 시그니처 통일하되, `prev_/curr_afterstate`는 미사용. 내부
  `_pending_state/action`로 (s,a) 추적.
- **결정**: 비종료 갱신은 다음 `choose_action` 시작부에서 (s_{t+1}을 보고 max bootstrap),
  종료 갱신은 `observe(done=True)`에서 즉시.
- **결정**: `learn_from_exploration` 플래그 없음 (Q-learning의 max가 이미 off-policy 성격).
- **결정**: 평가 모드에선 _pending 갱신도 스킵 (학습용 _pending이 평가에 의해 오염되지 않도록).
- self-test 6개 모두 통과 (2000판 학습 후 vs Random 93.6%).

## Task 9: `experiments/ex5_q_learning.py`
- TDAgent vs QAgent 동일 하이퍼로 학습 비교.
- 결과 (5000판):
  - **H1.5a 부분 지지**: TD 초기 0.91, Q 초기 0.85 — Q가 약간 뒤짐. 후반 비슷 (0.97 vs 0.95).
    이유: Q는 (s,a) 키라 ~9× 더 큰 상태공간 → sample efficiency 열위.
  - **H1.5b 지지**: 진영 비대칭 발견 → X 진영만 비교 시 Q 0.95 ≈ TD 0.99 (사실상 동치).
  - **H1.5c 지지**: TD 0.97, Q 0.95 모두 90%+.
- **이슈 노트**: 처음엔 Q-O가 TD-X에게 0% 승률이라 H1.5b 부정으로 보였으나, 둘 다 X로만
  학습됐기 때문임을 깨닫고 같은 진영 비교로 수정.

## Task 10: `README.md` 정리 + 종합 보고서
- 진행 현황 체크 / 자료 링크 / 가설 결과 요약표 / 흥미로운 발견 / 실행 방법 정리.
- 모든 모듈 self-test 통과, 모든 실험 PNG 16개 생성 확인.

---

## 종합

- 총 15개 검증 가설 중: ✅ 11개 지지, △ 3개 부분 지지, ❌ 1개 부정 (H1.4c).
- 부정/부분 지지의 원인은 모두 **틱택토 환경의 단순성** (짧은 에피소드, deterministic
  transitions, 작은 상태공간, 강한 학습 신호)으로 추정. 다른 도메인에서 재검증 필요.
- 모든 코드는 자체 self-test 포함, 헤드리스 환경에서 실행 가능, 시드 고정으로 재현성 보장.
