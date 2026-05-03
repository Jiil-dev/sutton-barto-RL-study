"""TD(0) 학습 에이전트 (책 §1.5의 알고리즘).

가치 함수 V(s)를 학습하며, afterstate 방식으로 행동 선택.
- afterstate: 에이전트가 행동을 둔 직후의 상태
- "각 합법 수의 afterstate 가치"를 비교해서 가장 높은 곳에 둠

TD 갱신 규칙 (탐욕 수 직후만):
    V(prev_afterstate) ← V(prev_afterstate) + α × [V(curr_afterstate) - V(prev_afterstate)]

Reference: Sutton & Barto, RL: An Introduction (2nd Ed.), §1.5
"""

from __future__ import annotations
import random
from typing import TYPE_CHECKING

# 순환 import 방지용 패턴: 타입 힌트에만 쓰는 import는 TYPE_CHECKING 안에
# (실제 런타임엔 import 안 됨)
if TYPE_CHECKING:
    from env import TicTacToeEnv

State = tuple[int, ...]


class TDAgent:
    """TD(0) 가치 함수 학습 에이전트.

    사용 예:
        agent = TDAgent(alpha=0.1, epsilon=0.1)
        # 학습 루프에서:
        action, was_exploratory = agent.choose_action(state, legal, env)
        # ... env.step() ...
        agent.observe(prev_afterstate, curr_afterstate, reward, done, was_exploratory)

    Attributes:
        alpha: 학습률 (step-size). 0~1.
        epsilon: 탐험 확률. 0이면 탐욕만.
        default_value: 처음 보는 상태의 기본 가치.
        learn_from_exploration: True면 탐험 수에서도 학습 (Ex 1.4 비교용).
        training: 학습 모드 토글. False면 ε=0이고 갱신 없음.
        values: 학습된 가치 함수 (state → value).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        default_value: float = 0.0,
        learn_from_exploration: bool = False,
    ):
        # 하이퍼파라미터
        self.alpha = alpha
        self.epsilon = epsilon
        self.default_value = default_value
        self.learn_from_exploration = learn_from_exploration

        # 모드 토글 (학습 시 True, 평가 시 False)
        self.training = True

        # 가치 함수: 본 적 있는 상태만 dict에 저장
        # 처음 보는 상태는 get_value()에서 default_value 반환
        self.values: dict[State, float] = {}

    # === 가치 함수 접근 ===

    def get_value(self, state: State) -> float:
        """상태의 현재 가치를 반환. 본 적 없으면 default_value.

        dict.get(key, default) 패턴으로 한 줄 처리.
        """
        return self.values.get(state, self.default_value)

    # === 행동 선택 (정책) ===

    def choose_action(
        self,
        state: State,
        legal_actions: list[int],
        env: "TicTacToeEnv",
    ) -> tuple[int, bool]:
        """ε-greedy로 행동 선택. 선택한 행동과 탐험 여부를 반환.

        afterstate 방식:
        각 합법 수에 대해 "두면 갈 상태"를 시뮬레이션하고,
        그 상태들의 V값을 비교해서 가장 높은 곳에 둠.

        Args:
            state: 현재 상태 (행동 선택에 직접 안 쓰지만 일관성 위해).
            legal_actions: 둘 수 있는 위치 리스트.
            env: 환경 (simulate_step 호출용).

        Returns:
            (action, is_exploratory)
            - action: 선택한 위치 (0~8).
            - is_exploratory: 탐험 수면 True, 탐욕 수면 False.
              평가 모드에서는 항상 False.
        """
        # 평가 모드면 무조건 탐욕
        effective_epsilon = self.epsilon if self.training else 0.0

        # ε 확률로 탐험
        if random.random() < effective_epsilon:
            return random.choice(legal_actions), True

        # 탐욕: 각 행동의 afterstate 가치 계산해서 argmax
        # (여러 개가 동률일 때 랜덤 선택 → 학습 초기에 한쪽으로 편향 방지)
        best_value = -float("inf")
        best_actions: list[int] = []

        for a in legal_actions:
            afterstate = env.simulate_step(a)
            value = self.get_value(afterstate)

            if value > best_value:
                best_value = value
                best_actions = [a]
            elif value == best_value:
                best_actions.append(a)

        return random.choice(best_actions), False

    # === 학습 (TD 갱신) ===

    def observe(
        self,
        prev_afterstate: State | None,
        curr_afterstate: State,
        reward: float,
        done: bool,
        was_exploratory: bool,
    ) -> None:
        """한 transition을 관찰하고 V를 갱신.

        호출 시점: env.step() 직후, "에이전트가 직접 둔 수 뒤"에서.

        Args:
            prev_afterstate: 직전에 우리가 둔 후의 상태. 게임 첫 수면 None.
            curr_afterstate: 방금 우리가 둔 후의 상태.
            reward: 이번 수로 받은 즉각 보상.
            done: 게임 종료 여부.
            was_exploratory: 이번 수가 탐험이었는지.
        """
        # 평가 모드면 학습 안 함
        if not self.training:
            return

        # 1) 종료 상태의 가치는 보상 그대로 (고정값)
        #    승리시 +1, 패배시 -1, 무승부시 0
        #    이 값은 dict에 저장해두면 다음 게임에서 backup의 타깃이 됨
        if done:
            self.values[curr_afterstate] = reward

        # 2) 탐험 수에서는 (기본 설정상) 갱신 안 함
        #    learn_from_exploration=True면 갱신함 (Ex 1.4 비교용)
        if was_exploratory and not self.learn_from_exploration:
            # 갱신 스킵 — 단, 종료 상태 저장은 위에서 이미 했음
            return

        # 3) 첫 수면 prev가 없으니 갱신 불가
        if prev_afterstate is None:
            return

        # 4) TD 갱신:
        #    V(prev) ← V(prev) + α × [V(curr) - V(prev)]
        prev_v = self.get_value(prev_afterstate)
        curr_v = self.get_value(curr_afterstate)
        td_error = curr_v - prev_v
        self.values[prev_afterstate] = prev_v + self.alpha * td_error

    # === 유틸리티 ===

    def num_known_states(self) -> int:
        """학습 중 본 적 있는 상태 개수. 학습 진척도 확인용."""
        return len(self.values)


# === 자가 테스트 ===
if __name__ == "__main__":
    # env.py를 같은 디렉토리에서 import 가능하도록 경로 조정
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from env import TicTacToeEnv, PLAYER_X, PLAYER_O

    print("=" * 40)
    print("TDAgent 단위 테스트")
    print("=" * 40)

    agent = TDAgent(alpha=0.1, epsilon=0.1)
    env = TicTacToeEnv()
    state = env.reset()

    # 1) 빈 보드에서 행동 선택이 되는가
    action, was_exp = agent.choose_action(state, env.legal_actions(), env)
    print(f"빈 보드에서 선택한 수: {action} (탐험: {was_exp})")
    assert action in range(9), "유효한 수여야 함"

    # 2) 한 수 두고 V 갱신이 되는가
    prev_after = env.simulate_step(action)
    next_state, reward, done, info = env.step(action)
    # 상대도 한 수 둔다고 가정 (그냥 무작위)
    if not done:
        opp_action = random.choice(env.legal_actions())
        next_state, reward, done, info = env.step(opp_action)

    # 우리 다음 수
    if not done:
        action2, was_exp2 = agent.choose_action(next_state, env.legal_actions(), env)
        curr_after = env.simulate_step(action2)
        agent.observe(prev_after, curr_after, 0.0, False, was_exp2)
        print(f"V 갱신 후 알고 있는 상태 수: {agent.num_known_states()}")

    # 3) 학습 모드 토글
    agent.training = False
    action3, was_exp3 = agent.choose_action(state, [0, 1, 2, 3], env)
    assert was_exp3 is False, "평가 모드에선 탐험 안 함"
    print(f"평가 모드 선택: {action3} (탐험: {was_exp3} — 항상 False여야 함)")

    print("\n✅ 모든 테스트 통과")