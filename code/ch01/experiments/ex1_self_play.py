"""Exercise 1.1: Self-Play 검증 실험.

검증할 가설 (`notes/ch01/exercises.md` 참고):
- H1.1a: 자가 대국 학습이 진행될수록 무승부 비율이 증가한다.
- H1.1b: 무작위 상대로 학습한 정책과 자가 대국으로 학습한 정책은 행동이 다르다
        (예: 첫 수 분포, 평균 게임 길이가 다름).
- H1.1c: 자가 대국 정책을 무작위 상대로 평가하면 잘 이기지만, 무작위 상대로 학습한
        정책보다 약간 보수적일 수 있다.

실험 설계:
1. 두 시나리오 학습 (동일 하이퍼파라미터, 동일 시드, 동일 num_games):
   A) `td_vs_random`: TD-X vs RandomAgent (한쪽만 학습)
   B) `td_self_play`: TD-X vs TD-O (양쪽 모두 학습, 별도 인스턴스 두 개)
2. 학습 중: 자가 대국의 누적 결과로 슬라이딩 윈도우 무승부 비율 기록 (H1.1a).
3. 학습 후: 두 시나리오의 X 측 정책에 대해
   a) 빈 보드 첫 수 분포 (H1.1b)
   b) RandomAgent 상대 평가 (H1.1c)
4. 산출물: 결과 PNG들 → results/ch01/ex1_*.png
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# experiments/ → code/ch01/ 을 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from env import PLAYER_X, TicTacToeEnv
from train import play_game, train, evaluate
from utils import (
    ensure_results_dir,
    first_move_distribution,
    plot_first_move_heatmap,
    plot_learning_curves,
    plot_outcome_distribution,
    set_seed,
)

# === 실험 하이퍼파라미터 ===
NUM_GAMES = 5000          # 학습 게임 수
ALPHA = 0.2               # 학습률
EPSILON = 0.1             # 탐험 확률
EVAL_EVERY = 250          # 학습 곡선 평가 주기
EVAL_GAMES = 300          # 평가 1회당 게임 수
WINDOW = 200              # 자가 대국 무승부율 슬라이딩 윈도우
SEED = 42


def run_td_vs_random() -> tuple[TDAgent, dict]:
    """A 시나리오: TD-X 학습 (vs RandomAgent O)."""
    td_x = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    rand_o = RandomAgent()
    history = train(
        td_x, rand_o,
        num_games=NUM_GAMES,
        eval_every=EVAL_EVERY,
        eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(),
        eval_target="x",
    )
    return td_x, history


def run_td_self_play() -> tuple[TDAgent, TDAgent, dict]:
    """B 시나리오: TD-X vs TD-O 자가 대국. 별도 인스턴스 2개로 values dict 격리."""
    td_x = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    td_o = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    history = train(
        td_x, td_o,
        num_games=NUM_GAMES,
        eval_every=EVAL_EVERY,
        eval_games=EVAL_GAMES,
        # 자가 대국 중에도 학습 곡선 비교를 위해 동일하게 RandomAgent 상대로 평가.
        eval_opponent_factory=lambda: RandomAgent(),
        eval_target="x",
    )
    return td_x, td_o, history


def sliding_draw_rate(results: list[int | None], window: int) -> tuple[list[int], list[float]]:
    """슬라이딩 윈도우 무승부 비율 계산.

    Returns:
        (윈도우 우측 끝 인덱스 리스트, 같은 길이의 무승부 비율 리스트)
    """
    xs: list[int] = []
    ys: list[float] = []
    draws_in_window = 0
    from collections import deque
    dq: deque = deque()
    for i, r in enumerate(results):
        is_draw = 1 if r is None else 0
        dq.append(is_draw)
        draws_in_window += is_draw
        if len(dq) > window:
            draws_in_window -= dq.popleft()
        if len(dq) == window:
            xs.append(i + 1)
            ys.append(draws_in_window / window)
    return xs, ys


def plot_self_play_draw_rate(
    xs: list[int],
    ys: list[float],
    save_path: Path,
    window: int,
) -> None:
    """자가 대국 슬라이딩 무승부 비율 곡선."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#264653", linewidth=2)
    ax.set_xlabel("Self-play game index")
    ax.set_ylabel(f"Draw rate (window={window})")
    ax.set_title("H1.1a: Self-play draw rate over training")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    results_dir = ensure_results_dir()

    print(f"[ex1] 학습 시작 (num_games={NUM_GAMES}, alpha={ALPHA}, epsilon={EPSILON})")
    td_a, history_a = run_td_vs_random()
    print(f"  A) TD vs Random 완료. 알고있는 상태: {td_a.num_known_states()}")

    td_b_x, td_b_o, history_b = run_td_self_play()
    print(f"  B) TD self-play 완료. X상태:{td_b_x.num_known_states()} O상태:{td_b_o.num_known_states()}")

    # === H1.1a: 자가 대국 무승부 비율 ===
    xs, ys = sliding_draw_rate(history_b["results"], window=WINDOW)
    plot_self_play_draw_rate(
        xs, ys, results_dir / "ex1_h1a_self_play_draw_rate.png", WINDOW
    )
    if ys:
        first_quarter = sum(ys[: len(ys) // 4]) / max(1, len(ys) // 4)
        last_quarter = sum(ys[-len(ys) // 4:]) / max(1, len(ys) // 4)
        print(f"[H1.1a] 무승부 비율 (윈도우 {WINDOW}): 초반 {first_quarter:.3f} → 후반 {last_quarter:.3f}")
        if last_quarter > first_quarter:
            print("  ✅ 무승부 비율이 증가 추세 (가설 H1.1a 지지)")
        else:
            print("  ⚠️ 증가 추세가 명확하지 않음 — 더 긴 학습 또는 다른 하이퍼 필요할 수 있음")

    # === H1.1b: 두 정책의 첫 수 분포 비교 ===
    env = TicTacToeEnv()
    dist_a = first_move_distribution(td_a, env, n_samples=2000)
    dist_b = first_move_distribution(td_b_x, env, n_samples=2000)
    plot_first_move_heatmap(dist_a, "A) TD vs Random — first move", results_dir / "ex1_h1b_first_move_A.png")
    plot_first_move_heatmap(dist_b, "B) TD self-play — first move", results_dir / "ex1_h1b_first_move_B.png")
    # 분포 차이를 정량화: L1 거리.
    l1 = sum(abs(dist_a[k] - dist_b[k]) for k in range(9))
    print(f"[H1.1b] 첫 수 분포 L1 거리: {l1:.3f}")
    if l1 > 0.2:
        print("  ✅ 두 정책의 행동이 명확히 다름 (가설 H1.1b 지지)")
    else:
        print("  ⚠️ 차이가 작음 — 두 정책이 비슷한 수렴점에 도달했을 가능성")

    # === H1.1c: 두 정책을 RandomAgent로 평가 ===
    eval_a = evaluate(td_a, lambda: RandomAgent(), num_games=1000)
    eval_b = evaluate(td_b_x, lambda: RandomAgent(), num_games=1000)
    print(f"[H1.1c] vs Random 1000판:")
    print(f"  A) TD-vs-Random:  {eval_a}")
    print(f"  B) TD-self-play:  {eval_b}")
    plot_outcome_distribution(
        {"A) TD vs Random": eval_a, "B) TD self-play": eval_b},
        "H1.1c: vs Random opponent (after training)",
        results_dir / "ex1_h1c_vs_random.png",
    )
    if eval_a["wins"] > eval_b["wins"]:
        print("  ✅ TD-vs-Random 정책이 무작위 상대에게 더 강함 (자가 대국 정책이 더 보수적임 — H1.1c 지지)")
    else:
        print("  ⚠️ 자가 대국 정책이 더 강하거나 비슷 — 가설 재검토 필요")

    # 학습 곡선도 같이 저장 (참고용)
    plot_learning_curves(
        {"A) TD vs Random": history_a, "B) TD self-play (X)": history_b},
        "Learning curves (vs Random eval)",
        results_dir / "ex1_learning_curves.png",
    )

    print(f"\n[ex1] 완료. 산출물: {results_dir}/ex1_*.png")


if __name__ == "__main__":
    main()
