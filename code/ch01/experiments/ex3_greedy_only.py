"""Exercise 1.3: Greedy-only player 검증 실험.

검증할 가설:
- H1.3a: 같은 게임 수 학습 후 평가 시, ε-greedy 학습 정책이 greedy-only(ε=0)보다 승률 높다.
- H1.3b: ε 값에 따른 trade-off가 존재한다 (너무 크면 학습 불안정, 너무 작으면 탐험 부족).
- H1.3c: 학습 후 평가는 ε=0으로 해야 진짜 실력 측정 가능.

실험 설계:
1. ε ∈ {0.0, 0.05, 0.1, 0.2, 0.4} 각각으로 TD agent 학습 (vs RandomAgent, 동일 게임 수).
2. 각 정책을 ε=0으로 frozen 평가 → win rate 비교 (H1.3a, H1.3b).
3. ε=0.1 정책 하나를 골라 ε=0 평가 vs ε=0.1(노이즈 유지) 평가 두 방식 비교 (H1.3c).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from env import TicTacToeEnv
from train import evaluate, train
from utils import (
    ensure_results_dir,
    plot_learning_curves,
    plot_outcome_distribution,
    set_seed,
)

NUM_GAMES = 4000
ALPHA = 0.2
EVAL_EVERY = 200
EVAL_GAMES = 300
SEED = 11
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.4]


def train_with_epsilon(epsilon: float) -> tuple[TDAgent, dict]:
    """주어진 ε로 학습 후 (agent, history) 반환. 시드는 매 호출마다 재설정해
    공정 비교 (모든 정책이 동일한 RNG 시퀀스로 시작)."""
    set_seed(SEED)
    agent = TDAgent(alpha=ALPHA, epsilon=epsilon)
    history = train(
        agent, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )
    return agent, history


def evaluate_with_eval_epsilon(
    agent: TDAgent,
    eval_epsilon: float,
    num_games: int,
) -> dict[str, float]:
    """평가 시점의 ε을 임시로 바꿔 평가.

    훈련용 ε와 평가용 ε을 분리하기 위해 self.epsilon을 잠시 덮어쓴다.
    학습 모드(self.training)도 함께 켜야 ε이 실제로 적용된다는 점 주의:
    TDAgent.choose_action은 training=False면 effective_epsilon=0으로 강제하기 때문.
    여기선 "평가는 학습 갱신 없이, 단지 ε만 적용"이 목적이므로
    learn=False인 경로(=evaluate)를 흉내내야 하지만 ε 노이즈는 살려야 한다.
    → training=True로 두고, evaluate()를 쓰지 않고 직접 play_game(learn=False)로 진행.
    """
    from train import play_game

    original_eps = agent.epsilon
    original_train = agent.training

    agent.epsilon = eval_epsilon
    agent.training = True  # ε이 0이 아니어야 의미; learn=False라 갱신은 안 됨
    env = TicTacToeEnv()

    wins = losses = draws = 0
    try:
        for _ in range(num_games):
            opponent = RandomAgent()
            opponent.training = False
            winner = play_game(env, agent, opponent, learn=False)
            if winner == 1:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
    finally:
        agent.epsilon = original_eps
        agent.training = original_train

    return {
        "wins": wins / num_games,
        "losses": losses / num_games,
        "draws": draws / num_games,
    }


def main() -> None:
    results_dir = ensure_results_dir()

    # === Phase 1: ε별 학습 ===
    print(f"[ex3] ε별 학습 (num_games={NUM_GAMES})")
    agents: dict[float, TDAgent] = {}
    histories: dict[str, dict] = {}
    finals: dict[str, dict[str, float]] = {}

    for eps in EPSILONS:
        agent, history = train_with_epsilon(eps)
        agents[eps] = agent
        label = f"ε={eps}"
        histories[label] = history
        # frozen 평가 (training=False, ε=0)
        final = evaluate(agent, lambda: RandomAgent(), num_games=1000)
        finals[label] = final
        print(f"  ε={eps}: states={agent.num_known_states():4d}, frozen vs Random: {final}")

    # H1.3a/b: 학습 곡선 + 최종 승률 막대
    plot_learning_curves(
        histories,
        "H1.3a/b: Learning curve by ε (eval frozen vs Random)",
        results_dir / "ex3_h1ab_curves.png",
    )
    plot_outcome_distribution(
        finals,
        "H1.3a/b: Final frozen evaluation by ε",
        results_dir / "ex3_h1ab_finals.png",
    )

    # H1.3a 판정: ε=0이 다른 ε들보다 승률 낮은가?
    win_zero = finals["ε=0.0"]["wins"]
    win_others = [v["wins"] for k, v in finals.items() if k != "ε=0.0"]
    best_other = max(win_others)
    print(f"[H1.3a] ε=0 win={win_zero:.3f}, 최고 ε>0 win={best_other:.3f}")
    if best_other > win_zero + 0.02:
        print("  ✅ 탐험이 있는 정책이 더 강함 (H1.3a 지지)")
    else:
        print("  ⚠️ ε=0이 비슷하거나 더 강함 — 운 좋게 최적 경로 잡았을 가능성")

    # H1.3b: ε에 따른 trade-off (정성)
    print("[H1.3b] ε별 win rate trade-off:")
    for eps in EPSILONS:
        print(f"  ε={eps}: {finals[f'ε={eps}']['wins']:.3f}")

    # === Phase 2: H1.3c — 평가 모드 비교 ===
    print("\n[ex3] Phase 2: 평가 모드 비교 (ε=0.1로 학습한 정책)")
    agent_for_c = agents[0.1]
    eval_frozen = evaluate(agent_for_c, lambda: RandomAgent(), num_games=1000)
    eval_noisy = evaluate_with_eval_epsilon(agent_for_c, eval_epsilon=0.1, num_games=1000)
    print(f"  평가 ε=0   (frozen):  {eval_frozen}")
    print(f"  평가 ε=0.1 (탐험 켜짐): {eval_noisy}")
    plot_outcome_distribution(
        {"eval ε=0 (frozen)": eval_frozen, "eval ε=0.1 (noisy)": eval_noisy},
        "H1.3c: Effect of evaluation-time ε (same agent)",
        results_dir / "ex3_h1c_eval_mode.png",
    )
    if eval_frozen["wins"] > eval_noisy["wins"]:
        print("  ✅ frozen 평가가 더 높은 승률 — 평가 시 ε=0이 진짜 실력 (H1.3c 지지)")
    else:
        print("  ⚠️ noisy가 더 강함 — 학습된 V가 충분히 안정되지 않았을 수 있음")

    print(f"\n[ex3] 완료. 산출물: {results_dir}/ex3_*.png")


if __name__ == "__main__":
    main()
