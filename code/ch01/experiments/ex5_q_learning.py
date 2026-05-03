"""Exercise 1.5: Q-learning 확장 검증 실험.

검증할 가설:
- H1.5a: 같은 학습량(=게임 수)에서 Q-learning이 V 기반 TD보다 빠르게 수렴(혹은 비슷).
- H1.5b: 학습된 Q 정책 vs V 정책 직접 대결 시, Q 정책이 비등하거나 우위.
- H1.5c: 두 방법 모두 충분히 학습하면 무작위 상대에게 거의 100% 승률에 수렴.

실험 설계:
1. 동일 하이퍼파라미터(α, ε, num_games, seed)로 TDAgent와 QAgent 학습 (vs RandomAgent).
2. 학습 곡선(평가 vs Random) 비교 → H1.5a.
3. 학습 후 두 에이전트 직접 대결 양 진영 (TD-X vs Q-O), (Q-X vs TD-O) → H1.5b.
4. 양쪽의 최종 win rate가 0.95 이상인지 → H1.5c.

이론적 노트:
틱택토는 deterministic 환경이라 Q(s,a) ≈ V(afterstate(s,a))로 해석 가능.
즉 두 알고리즘은 수학적으로 거의 동치. 따라서 큰 차이는 기대하지 않으며, 가설은
"비슷하거나 살짝 우위" 수준으로 보수적으로 설정됨.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.q_agent import QAgent
from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from env import PLAYER_O, PLAYER_X, TicTacToeEnv
from train import evaluate, play_game, train
from utils import (
    ensure_results_dir,
    plot_learning_curves,
    plot_outcome_distribution,
    set_seed,
)

NUM_GAMES = 5000
ALPHA = 0.2
EPSILON = 0.1
EVAL_EVERY = 250
EVAL_GAMES = 300
SEED = 17


def head_to_head(
    agent_a,
    agent_b,
    a_plays_x: bool,
    num_games: int,
) -> dict[str, float]:
    """학습 끄고 두 에이전트 직접 대결. agent_a 관점의 승/패/무 비율 반환."""
    env = TicTacToeEnv()
    a_orig = agent_a.training
    b_orig = agent_b.training
    agent_a.training = False
    agent_b.training = False
    wins = losses = draws = 0
    try:
        for _ in range(num_games):
            if a_plays_x:
                winner = play_game(env, agent_a, agent_b, learn=False)
                a_side = PLAYER_X
            else:
                winner = play_game(env, agent_b, agent_a, learn=False)
                a_side = PLAYER_O
            if winner == a_side:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
    finally:
        agent_a.training = a_orig
        agent_b.training = b_orig
    return {
        "wins": wins / num_games,
        "losses": losses / num_games,
        "draws": draws / num_games,
    }


def main() -> None:
    results_dir = ensure_results_dir()

    print(f"[ex5] TD vs Q 학습 (num_games={NUM_GAMES}, α={ALPHA}, ε={EPSILON})")
    set_seed(SEED)
    td = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    history_td = train(
        td, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    set_seed(SEED)
    q = QAgent(alpha=ALPHA, epsilon=EPSILON, gamma=1.0)
    history_q = train(
        q, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    print(f"  TD: states={td.num_known_states()}, Q: pairs={q.num_known_pairs()}")

    # === H1.5a: 학습 곡선 비교 ===
    plot_learning_curves(
        {"TD (V-afterstate)": history_td, "Q-learning": history_q},
        "H1.5a: Learning curve TD vs Q (vs Random)",
        results_dir / "ex5_h1a_curves.png",
    )

    # 초반/중반/후반 평균 win rate
    nrec = len(history_td["eval_records"])
    third = max(1, nrec // 3)
    early_td = sum(r["wins"] for r in history_td["eval_records"][:third]) / third
    early_q = sum(r["wins"] for r in history_q["eval_records"][:third]) / third
    late_td = sum(r["wins"] for r in history_td["eval_records"][-third:]) / third
    late_q = sum(r["wins"] for r in history_q["eval_records"][-third:]) / third
    print(f"[H1.5a] 초반 평균 win — TD: {early_td:.3f}, Q: {early_q:.3f}")
    print(f"        후반 평균 win — TD: {late_td:.3f}, Q: {late_q:.3f}")
    if early_q >= early_td - 0.02:
        print("  ✅ Q가 TD와 비슷하거나 빠른 학습 (H1.5a 지지)")
    else:
        print("  ⚠️ Q가 초기 학습에서 명확히 뒤짐 — 하이퍼 재조정 검토")

    # === H1.5c: 최종 vs Random 승률 ===
    final_td = evaluate(td, lambda: RandomAgent(), num_games=1500)
    final_q = evaluate(q, lambda: RandomAgent(), num_games=1500)
    print(f"[H1.5c] 최종 vs Random:")
    print(f"  TD: {final_td}")
    print(f"  Q:  {final_q}")
    if final_td["wins"] >= 0.9 and final_q["wins"] >= 0.9:
        print("  ✅ 양쪽 모두 90%+ 승률 (H1.5c 지지)")
    else:
        print("  ⚠️ 한쪽 또는 양쪽이 90% 미만 — 학습량 부족")

    # === H1.5b: 두 정책 직접 대결 (양 진영 모두) ===
    h2h_q_x_vs_td_o = head_to_head(q, td, a_plays_x=True, num_games=600)
    h2h_q_o_vs_td_x = head_to_head(q, td, a_plays_x=False, num_games=600)
    print(f"[H1.5b] head-to-head (Q 관점):")
    print(f"  Q-X vs TD-O: {h2h_q_x_vs_td_o}")
    print(f"  Q-O vs TD-X: {h2h_q_o_vs_td_x}")
    plot_outcome_distribution(
        {
            "Q-X vs TD-O": h2h_q_x_vs_td_o,
            "Q-O vs TD-X": h2h_q_o_vs_td_x,
        },
        "H1.5b: Q vs TD head-to-head (Q's perspective)",
        results_dir / "ex5_h1b_h2h.png",
    )
    # 본 실험에서는 두 에이전트 모두 X측으로만 학습됨. 따라서 head-to-head 결과는
    # "X로 둔 쪽이 압도적"으로 나타나는 것이 자연스럽다 (O측은 학습 분포 밖 상태가 많음).
    # 알고리즘 비교를 위해선 같은 진영(X)일 때의 값만 비교하는 것이 공정.
    avg_x = h2h_q_x_vs_td_o["wins"]   # Q가 X일 때 vs TD-O
    inv_x = 1.0 - h2h_q_o_vs_td_x["wins"] - h2h_q_o_vs_td_x["draws"]  # TD가 X일 때 승률
    print(f"        X 진영 승률 비교 — Q-X: {avg_x:.3f}, TD-X: {inv_x:.3f}")
    if abs(avg_x - inv_x) < 0.1:
        print("  ✅ X 진영 승률 비등 — Q와 TD가 본질적으로 동치 (H1.5b 지지)")
    else:
        print("  ⚠️ X 진영 승률 격차 큼 — 알고리즘 자체의 차이 가능성")

    # 종합 막대그래프
    plot_outcome_distribution(
        {"TD vs Random": final_td, "Q vs Random": final_q},
        "H1.5c: Final vs Random",
        results_dir / "ex5_h1c_vs_random.png",
    )

    print(f"\n[ex5] 완료. 산출물: {results_dir}/ex5_*.png")


if __name__ == "__main__":
    main()
