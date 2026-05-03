"""무작위 정책 에이전트.

학습하지 않고, 매 수마다 합법 수 중 하나를 균등 무작위로 선택한다.
주된 용도:
    1) 학습 에이전트(TDAgent 등)의 **상대(opponent)**
    2) 평가 시 **베이스라인** (학습 진척도 측정용)
    3) 인터페이스 일관성 검증 — 학습 루프가 어떤 에이전트든 동일하게 다룰 수 있는지

설계 메모:
- TDAgent와 동일한 시그니처(choose_action, observe, training)를 갖는다.
- 학습이 없으므로 observe()는 no-op이고 training 토글은 동작에 영향 없음.
- was_exploratory는 항상 False로 보고한다 (이유는 docstring 참조).
- 재현성은 에이전트 내부가 아닌 **실험 스크립트의 random.seed()** 로 제어한다
  (모듈 레벨 random을 사용해 TDAgent와 동일한 RNG에 맞춤).

Reference: Sutton & Barto, RL: An Introduction (2nd Ed.), §1.5
"""

from __future__ import annotations
import random
from typing import TYPE_CHECKING

# TDAgent와 동일한 패턴: 타입 힌트용 import는 런타임에 로드되지 않도록.
if TYPE_CHECKING:
    from env import TicTacToeEnv

State = tuple[int, ...]


class RandomAgent:
    """합법 수 중 균등 무작위로 선택하는 에이전트 (학습 없음).

    사용 예:
        agent = RandomAgent()
        action, was_exploratory = agent.choose_action(state, legal, env)
        # observe(...) 호출은 인터페이스 통일을 위해 가능하지만 아무것도 안 함.

    Attributes:
        training: 학습 모드 토글. RandomAgent는 학습이 없으므로 동작에 영향 없음.
                  TDAgent 등과 인터페이스를 통일하기 위한 placeholder.
    """

    def __init__(self):
        # 인터페이스 통일을 위한 placeholder. 동작에는 영향 없음.
        self.training = True

    def choose_action(
        self,
        state: State,
        legal_actions: list[int],
        env: "TicTacToeEnv",
    ) -> tuple[int, bool]:
        """합법 수 중 균등 무작위로 하나 선택.

        Args:
            state: 현재 상태. RandomAgent는 사용하지 않지만 인터페이스 통일을 위해 받음.
            legal_actions: 둘 수 있는 위치 리스트. 비어 있으면 안 됨 (게임이 끝나지 않았을 때만 호출).
            env: 환경. 사용하지 않지만 인터페이스 통일을 위해 받음.

        Returns:
            (action, is_exploratory)
            - action: 선택한 위치 (0~8).
            - is_exploratory: **항상 False**.

              근거: "탐험"은 "현재 정책의 탐욕 선택에서 의도적으로 벗어났다"는 의미인데,
              RandomAgent는 탐욕 정책 자체가 정의되지 않으므로 "탐험"이라는 개념이
              적용되지 않는다. False가 가장 안전한 기본값이며, 이 플래그는 자기 자신의
              observe()로 전달되어 학습 갱신을 좌우하는데 RandomAgent는 학습하지 않으므로
              값이 무엇이든 무관하다.
        """
        # state, env는 인터페이스 통일용. 본 에이전트는 사용 안 함.
        # _ 변수에 할당하지 않고 그냥 무시 (파이썬 관례).
        del state, env  # 미사용 인자임을 명시 (lint 경고 회피 + 의도 표현)

        return random.choice(legal_actions), False

    def observe(
        self,
        prev_afterstate: State | None,
        curr_afterstate: State,
        reward: float,
        done: bool,
        was_exploratory: bool,
    ) -> None:
        """no-op. RandomAgent는 학습하지 않는다.

        시그니처를 TDAgent와 동일하게 두어 학습 루프가 분기 없이 호출할 수 있게 한다.
        """
        # 모든 인자는 무시. 인터페이스 통일이 유일한 목적.
        del prev_afterstate, curr_afterstate, reward, done, was_exploratory


# === 자가 테스트 ===
if __name__ == "__main__":
    # env.py를 같은 디렉토리에서 import 가능하도록 경로 조정
    # (TDAgent와 동일한 패턴)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from env import TicTacToeEnv, PLAYER_X, PLAYER_O

    print("=" * 40)
    print("RandomAgent 단위 테스트")
    print("=" * 40)

    # 재현성 위한 시드 고정 (이 자가 테스트 안에서만)
    random.seed(0)

    # 1) choose_action 결과가 항상 합법 수인지 (1000회 샘플, 빈 보드에서)
    env = TicTacToeEnv()
    state = env.reset()  # 공개 API로 상태 획득
    agent = RandomAgent()

    for _ in range(1000):
        legal = env.legal_actions()
        action, was_exp = agent.choose_action(state, legal, env)
        assert action in legal, f"불법 수 선택: {action} (합법: {legal})"
        assert was_exp is False, "RandomAgent는 항상 was_exploratory=False여야 함"
    print("✅ 1000회 샘플: 모두 합법 수, was_exploratory=False")

    # 2) RandomAgent vs RandomAgent 한 게임을 끝까지 돌려서 예외 없이 종료되는지
    state = env.reset()
    agent_x = RandomAgent()
    agent_o = RandomAgent()
    move_count = 0
    while not env.done:
        current_agent = agent_x if env.current_player == PLAYER_X else agent_o
        action, _ = current_agent.choose_action(state, env.legal_actions(), env)
        state, _reward, _done, _info = env.step(action)
        move_count += 1
    # 틱택토는 최대 9수, 최소 5수에 끝남
    assert 5 <= move_count <= 9, f"수 횟수 비정상: {move_count}"
    print(f"✅ Random vs Random 한 게임 완주: {move_count}수, 승자={env.winner}")

    # 3) observe()가 어떤 인자를 받아도 예외 없이 통과하는지
    agent.observe(None, (0,) * 9, 0.0, False, False)
    agent.observe((0,) * 9, (1,) + (0,) * 8, 1.0, True, True)
    print("✅ observe() no-op: 다양한 인자 조합 통과")

    # 4) training 토글이 존재하고 동작에 영향 없음을 확인
    agent.training = False
    state = env.reset()
    legal = env.legal_actions()
    action_eval, was_exp_eval = agent.choose_action(state, legal, env)
    assert action_eval in legal, "training=False에서도 합법 수만 골라야 함"
    assert was_exp_eval is False, "training과 무관하게 항상 False"
    print("✅ training 토글: 동작에 영향 없음")

    # 5) 분포 sanity check: 빈 보드(9개 합법 수)에서 9000회 뽑으면 각 수가 ~1000회
    random.seed(0)
    state = env.reset()
    legal = env.legal_actions()  # 빈 보드라 항상 [0..8]
    counts = [0] * 9
    for _ in range(9000):
        action, _ = agent.choose_action(state, legal, env)
        counts[action] += 1
    print(f"✅ 9000회 분포 (이상치 ±20%): {counts}")
    for i, c in enumerate(counts):
        assert 800 <= c <= 1200, f"위치 {i}의 빈도가 비정상: {c}"

    print("\n✅ 모든 테스트 통과")
