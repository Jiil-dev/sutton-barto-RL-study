"""Exercise 1.4: Learning from exploration moves 검증 실험.

검증할 가설:
- H1.4a: 탐험에서 학습한 정책의 V값이 탐험에서 학습 안 한 정책보다 전반적으로 낮다
        (실제 행동 정책의 가치 = on-policy 가치는 탐험으로 인한 손실 반영).
- H1.4b: 두 정책을 ε=0으로 평가하면 "탐험에서 학습 안 한"(off-policy) 쪽이 승률 높다.
- H1.4c: 두 정책을 ε>0으로 평가하면 차이가 줄거나 역전될 수 있다 (on-policy가 자기 행동
        정책을 정확히 모델링했으므로 탐험 환경에선 더 robust).

실험 설계:
1. 동일 하이퍼파라미터(α, ε, num_games, seed) 두 TDAgent:
   - off-policy 류: learn_from_exploration=False (책의 기본 방식)
   - on-policy 류: learn_from_exploration=True
2. 학습 후 두 V 함수를 공통 상태(둘 다 본 적 있는 키)에 대해 비교 → H1.4a.
3. ε=0 frozen 평가 (H1.4b)와 ε=0.1 noisy 평가 (H1.4c) 양쪽 측정.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from env import TicTacToeEnv
from train import evaluate, play_game, train
from utils import (
    ensure_results_dir,
    plot_outcome_distribution,
    set_seed,
)

NUM_GAMES = 5000
ALPHA = 0.2
EPSILON = 0.2  # 탐험 효과를 더 명확히 보기 위해 약간 큰 값
EVAL_EVERY = 250
EVAL_GAMES = 300
SEED = 13


def evaluate_with_eval_epsilon(agent: TDAgent, eval_epsilon: float, num_games: int) -> dict:
    """평가 시 ε을 임시로 바꿔 측정 (학습 갱신 없음, 노이즈만 적용).
    ex3에서 정의한 함수와 동일 패턴이지만 import 의존성 줄이려고 여기 재정의.
    """
    original_eps = agent.epsilon
    original_train = agent.training
    agent.epsilon = eval_epsilon
    agent.training = True  # ε이 effective_epsilon에 반영되려면 training=True 필요
    env = TicTacToeEnv()
    wins = losses = draws = 0
    try:
        for _ in range(num_games):
            opp = RandomAgent()
            opp.training = False
            winner = play_game(env, agent, opp, learn=False)
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

    print(f"[ex4] 두 정책 학습 (num_games={NUM_GAMES}, ε={EPSILON})")
    set_seed(SEED)
    off_policy = TDAgent(alpha=ALPHA, epsilon=EPSILON, learn_from_exploration=False)
    train(
        off_policy, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    set_seed(SEED)
    on_policy = TDAgent(alpha=ALPHA, epsilon=EPSILON, learn_from_exploration=True)
    train(
        on_policy, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    print(f"  off-policy: {off_policy.num_known_states()} states")
    print(f"  on-policy:  {on_policy.num_known_states()} states")

    # === H1.4a: V 분포 비교 ===
    common_keys = set(off_policy.values.keys()) & set(on_policy.values.keys())
    if not common_keys:
        print("[H1.4a] 공통 상태 없음 — 학습 분기가 너무 다름. 비교 생략.")
    else:
        off_vals = [off_policy.values[k] for k in common_keys]
        on_vals = [on_policy.values[k] for k in common_keys]
        diff = [on_policy.values[k] - off_policy.values[k] for k in common_keys]
        print(f"[H1.4a] 공통 상태 {len(common_keys)}개:")
        print(f"  off-policy V 평균: {mean(off_vals):+.4f}")
        print(f"  on-policy  V 평균: {mean(on_vals):+.4f}")
        print(f"  (on - off) 평균:   {mean(diff):+.4f}")
        if mean(on_vals) < mean(off_vals):
            print("  ✅ on-policy V가 평균적으로 낮음 (탐험 손실 반영, H1.4a 지지)")
        else:
            print("  ⚠️ 예상과 반대 — 학습 충분치 않거나 탐험 영향 미미한 상태가 다수")

        # 분포 그림 (히스토그램)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(off_vals, bins=30, alpha=0.6, label="off-policy V (no exp learning)", color="#264653")
        ax.hist(on_vals, bins=30, alpha=0.6, label="on-policy V (learn from exp)", color="#e76f51")
        ax.set_xlabel("V(s)")
        ax.set_ylabel("count (common states)")
        ax.set_title("H1.4a: V distribution on common states")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "ex4_h1a_value_dist.png", dpi=120)
        plt.close(fig)

    # === H1.4b: ε=0 frozen 평가 ===
    eval_off_frozen = evaluate(off_policy, lambda: RandomAgent(), num_games=1500)
    eval_on_frozen = evaluate(on_policy, lambda: RandomAgent(), num_games=1500)
    print(f"[H1.4b] ε=0 frozen 평가:")
    print(f"  off-policy: {eval_off_frozen}")
    print(f"  on-policy:  {eval_on_frozen}")
    if eval_off_frozen["wins"] > eval_on_frozen["wins"]:
        print("  ✅ off-policy가 더 강함 (H1.4b 지지)")
    else:
        print("  ⚠️ on-policy가 같거나 더 강함 — 학습 노이즈 가능, 다른 시드에서 재확인 권장")

    # === H1.4c: ε=0.1 noisy 평가 ===
    eval_off_noisy = evaluate_with_eval_epsilon(off_policy, eval_epsilon=0.1, num_games=1500)
    eval_on_noisy = evaluate_with_eval_epsilon(on_policy, eval_epsilon=0.1, num_games=1500)
    print(f"[H1.4c] ε=0.1 noisy 평가:")
    print(f"  off-policy: {eval_off_noisy}")
    print(f"  on-policy:  {eval_on_noisy}")
    diff_frozen = eval_off_frozen["wins"] - eval_on_frozen["wins"]
    diff_noisy = eval_off_noisy["wins"] - eval_on_noisy["wins"]
    print(f"  격차 frozen: {diff_frozen:+.3f} → noisy: {diff_noisy:+.3f}")
    if diff_noisy < diff_frozen:
        print("  ✅ noisy 평가에서 격차 축소 또는 역전 (H1.4c 지지)")
    else:
        print("  ⚠️ 격차 변화 없음 또는 확대 — 가설 부분 부정 가능성")

    plot_outcome_distribution(
        {
            "off-policy (eval ε=0)": eval_off_frozen,
            "on-policy (eval ε=0)":  eval_on_frozen,
            "off-policy (eval ε=0.1)": eval_off_noisy,
            "on-policy (eval ε=0.1)":  eval_on_noisy,
        },
        "H1.4b/c: off-policy vs on-policy under different eval ε",
        results_dir / "ex4_h1bc_eval.png",
    )

    print(f"\n[ex4] 완료. 산출물: {results_dir}/ex4_*.png")


if __name__ == "__main__":
    main()
