"""Q-learning 에이전트 (책 §6.5의 알고리즘을 틱택토에 적용).

표준 Q-learning 갱신:
    Q(s_t, a_t) ← Q(s_t, a_t) + α [r_{t+1} + γ · max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
종료 상태에서는 bootstrap 항이 없고 target = r_{t+1}.

설계 메모:
- TDAgent와 인터페이스(choose_action / observe / training)를 통일하여 train.py에서
  분기 없이 사용 가능. 단, observe()의 prev_afterstate / curr_afterstate 인자는
  Q-learning에선 사용하지 않고, 내부 `_pending_state`/`_pending_action`으로 (s,a)를 추적.
- Q-갱신 타이밍 분리:
  * **종료 갱신** → observe(done=True)에서 즉시 (target=reward).
  * **비종료 갱신** → 다음 choose_action 시작부에서 (s_{t+1}을 보고 max bootstrap).
- `learn_from_exploration` 플래그 없음. Q-learning의 max 항이 이미 off-policy 성격을
  포함하므로, 탐험 행동에 대한 Q-갱신을 따로 막을 필요 없음 (오히려 막으면 표준에서 벗어남).
- Q는 dict[(state, action) → float]로 저장. 본 적 없는 키는 default_value(=0.0).

Reference: Sutton & Barto, RL: An Introduction (2nd Ed.), §6.5
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import TicTacToeEnv

State = tuple[int, ...]


class QAgent:
    """Q-learning 에이전트.

    Attributes:
        alpha, epsilon, gamma, default_value: 표준 하이퍼파라미터.
        training: 학습 모드 토글 (False면 ε=0이고 Q-갱신 없음).
        q: dict[(State, int), float] — Q 함수.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        default_value: float = 0.0,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.default_value = default_value
        self.training = True

        self.q: dict[tuple[State, int], float] = {}

        # 직전 (state, action)을 보관해 다음 choose_action에서 bootstrap 갱신에 사용.
        # observe(done=False)에서 reward를 채우고, 다음 choose_action에서 max를 계산해 갱신.
        # observe(done=True)에서는 즉시 종료 갱신을 적용하고 None으로 리셋.
        self._pending_state: State | None = None
        self._pending_action: int | None = None
        self._pending_reward: float = 0.0

    # === Q 접근 ===

    def get_q(self, state: State, action: int) -> float:
        return self.q.get((state, action), self.default_value)

    def num_known_pairs(self) -> int:
        """학습 중 본 (state, action) 쌍 개수."""
        return len(self.q)

    # === 행동 선택 (정책) ===

    def choose_action(
        self,
        state: State,
        legal_actions: list[int],
        env: "TicTacToeEnv",
    ) -> tuple[int, bool]:
        """ε-greedy로 행동 선택. 함수 시작부에서 직전 (s,a)에 대한 bootstrap 갱신 수행.

        Args:
            state: 현재 의사결정 상태 (이번에 선택할 base 상태).
            legal_actions: 둘 수 있는 위치.
            env: 인터페이스 통일용 (Q-learning은 simulate_step 불필요).

        Returns:
            (action, was_exploratory).
        """
        del env  # Q-learning은 (s,a) 평가에 simulate_step이 불필요.

        # === 직전 transition에 대한 비종료 Q-갱신 ===
        # 이 시점에 보이는 `state`는 직전 자기 행동 + 상대 행동 후의 상태,
        # 즉 Q(s_t, a_t) 갱신의 부트스트랩 대상이 되는 s_{t+1}이다.
        if self.training and self._pending_state is not None:
            # max_{a'} Q(state, a') — legal_actions가 비어있을 가능성 없음
            # (env.done이 아닐 때만 choose_action이 호출되므로 합법 수가 최소 1개).
            max_next = max(self.get_q(state, a) for a in legal_actions)
            target = self._pending_reward + self.gamma * max_next
            key = (self._pending_state, self._pending_action)
            cur = self.q.get(key, self.default_value)
            self.q[key] = cur + self.alpha * (target - cur)

        # === ε-greedy 선택 ===
        effective_epsilon = self.epsilon if self.training else 0.0
        if random.random() < effective_epsilon:
            action = random.choice(legal_actions)
            was_exp = True
        else:
            best_q = -float("inf")
            best_actions: list[int] = []
            for a in legal_actions:
                q = self.get_q(state, a)
                if q > best_q:
                    best_q = q
                    best_actions = [a]
                elif q == best_q:
                    best_actions.append(a)
            action = random.choice(best_actions)
            was_exp = False

        # === 다음 갱신용 pending 저장 ===
        # 평가 모드에선 pending을 갱신하지 않는다. 학습 후에 평가가 끼어들어도
        # 학습용 _pending이 오염되지 않도록.
        if self.training:
            self._pending_state = state
            self._pending_action = action

        return action, was_exp

    # === 학습 ===

    def observe(
        self,
        prev_afterstate: State | None,
        curr_afterstate: State,
        reward: float,
        done: bool,
        was_exploratory: bool,
    ) -> None:
        """Reward를 받고 종료 시 마지막 (s,a)에 대한 Q-갱신을 수행.

        prev_afterstate / curr_afterstate / was_exploratory는 인터페이스 통일을 위해
        받지만 Q-learning에선 사용하지 않는다 (직전 (s,a)는 내부 _pending로 추적).
        """
        del prev_afterstate, curr_afterstate, was_exploratory

        if not self.training:
            return

        if done:
            # 종료 상태에 대한 갱신: bootstrap 없이 reward 그대로가 target.
            if self._pending_state is not None:
                key = (self._pending_state, self._pending_action)
                cur = self.q.get(key, self.default_value)
                self.q[key] = cur + self.alpha * (reward - cur)
            # 다음 게임을 위해 pending 리셋.
            self._pending_state = None
            self._pending_action = None
            self._pending_reward = 0.0
        else:
            # 비종료: 다음 choose_action에서 부트스트랩에 사용할 reward 저장.
            self._pending_reward = reward


# === 자가 테스트 ===

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from env import EMPTY, PLAYER_O, PLAYER_X, TicTacToeEnv
    from agents.random_agent import RandomAgent
    from train import evaluate, play_game, train

    print("=" * 50)
    print("QAgent 자가 테스트")
    print("=" * 50)

    random.seed(0)

    # 1) 빈 보드에서 합법 수 선택
    env = TicTacToeEnv()
    state = env.reset()
    agent = QAgent(alpha=0.2, epsilon=0.1)
    action, was_exp = agent.choose_action(state, env.legal_actions(), env)
    assert action in env.legal_actions()
    print(f"[1] 빈 보드 선택: {action} (탐험: {was_exp})")

    # 2) 한 게임 후 Q 키가 늘어남
    play_game(env, agent, RandomAgent(), learn=True)
    n1 = agent.num_known_pairs()
    assert n1 > 0, "한 게임 후 Q 항목이 있어야 함"
    print(f"[2] 한 게임 후 (state,action) 쌍: {n1}")

    # 3) pending이 게임 종료 후 None으로 리셋되었는지
    assert agent._pending_state is None, "종료 후 pending이 None이어야 함"
    print("[3] 종료 후 pending 리셋 OK")

    # 4) 200판 학습 후 vs Random 평가에서 의미 있는 승률
    random.seed(0)
    agent2 = QAgent(alpha=0.3, epsilon=0.2)
    train(
        agent2, RandomAgent(),
        num_games=2000,
        eval_every=500, eval_games=200,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )
    rec = evaluate(agent2, lambda: RandomAgent(), num_games=500)
    print(f"[4] 2000판 학습 후 평가 (vs Random): {rec}")
    assert rec["wins"] >= 0.7, f"Q 학습 후 승률이 너무 낮음 (>=0.7 기대): {rec}"

    # 5) 평가 모드에서 ε=0이고 Q 갱신 없는지
    n_before = agent2.num_known_pairs()
    agent2.training = False
    for _ in range(50):
        play_game(env, agent2, RandomAgent(), learn=False)
    n_after = agent2.num_known_pairs()
    assert n_before == n_after, f"평가 모드에서 Q가 변경됨: {n_before} -> {n_after}"
    print(f"[5] 평가 모드에서 Q 동결 OK ({n_before} 유지)")

    # 6) eval 후 training 재개 시 stale pending이 학습을 망가뜨리지 않음
    agent2.training = True
    play_game(env, agent2, RandomAgent(), learn=True)
    print("[6] eval → training 전환 후 한 게임 정상 완주")

    print("\n✅ 모든 테스트 통과")
