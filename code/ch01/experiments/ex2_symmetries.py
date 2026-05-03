"""Exercise 1.2: Symmetries 검증 실험.

검증할 가설:
- H1.2a: 대칭성을 활용하면 같은 게임 수에서 학습 속도가 빨라진다.
- H1.2b: 상대가 대칭적인 정책(균등 무작위)일 때 대칭 활용은 손해 없다.
- H1.2c: 상대가 비대칭 정책일 때 대칭 활용 정책이 비활용 정책보다 승률이 낮다.

실험 설계:
1. 대칭 활용 에이전트 SymmetricTDAgent 정의 (TDAgent 상속, canonical_state 적용).
2. 비대칭 상대 BiasedRandomAgent 정의 (합법수 중 항상 최소 인덱스 선택 — 명백한 비대칭).
3. 시나리오:
   - vs RandomAgent (대칭 상대): Vanilla TD vs Symmetric TD 학습 곡선 비교 → H1.2a, H1.2b
   - vs BiasedRandomAgent: Vanilla TD vs Symmetric TD 평가 → H1.2c
4. 상태 공간 크기도 비교해 대칭성의 정량 효과 확인.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from env import PLAYER_X, TicTacToeEnv
from train import evaluate, train
from utils import (
    canonical_state,
    ensure_results_dir,
    plot_learning_curves,
    plot_outcome_distribution,
    set_seed,
)

State = tuple[int, ...]

NUM_GAMES = 4000
ALPHA = 0.2
EPSILON = 0.1
EVAL_EVERY = 200
EVAL_GAMES = 300
SEED = 7


# === 대칭 활용 TDAgent ===

class SymmetricTDAgent(TDAgent):
    """모든 상태를 정규형(8개 D4 대칭 중 사전식 최소)으로 축약하는 TDAgent.

    효과: 가치 dict의 키 공간이 ~1/8로 줄어 학습 속도가 빨라진다 (H1.2a).
    한계: 상대가 비대칭 패턴이면 같은 정규형으로 묶인 보드의 진짜 가치가 다를 수 있다 (H1.2c).

    구현: get_value와 observe에서 인자를 canonical_state로 정규화. choose_action은
    상속 그대로 (env.simulate_step 결과를 get_value에 넘기는 흐름은 정규화 후에도 동일).
    """

    def get_value(self, state: State) -> float:  # type: ignore[override]
        return super().get_value(canonical_state(state))

    def observe(  # type: ignore[override]
        self,
        prev_afterstate: State | None,
        curr_afterstate: State,
        reward: float,
        done: bool,
        was_exploratory: bool,
    ) -> None:
        prev_c = canonical_state(prev_afterstate) if prev_afterstate is not None else None
        curr_c = canonical_state(curr_afterstate)
        super().observe(prev_c, curr_c, reward, done, was_exploratory)


# === 비대칭 상대 ===

class BiasedRandomAgent:
    """합법 수 중 인덱스가 작을수록 더 큰 가중치로 뽑는 stochastic 비대칭 에이전트.

    설계 의도:
    - **결정론(BiasedRandom)**은 곧장 외워져 두 정책 모두 100% 승률로 수렴 → 차이 검출 불가.
    - 확률적이지만 명확한 비대칭이면, vanilla TD는 "위치 0에 두면 상대가 자주 X 옆에 둔다"는
      약점을 활용해 위치별 가치를 다르게 학습 가능. Symmetric TD는 회전/반사로 묶어버려
      위치별 약점을 볼 수 없음 → 같은 학습량에서 vanilla가 우위 기대.

    가중치: 인덱스 i에 대해 weight = 9 - i (높은 인덱스일수록 작은 가중치).
    """
    training = True  # placeholder

    def __init__(self):
        # __init__이 명시적이어야 instance 생성 시 확실히 호출됨 (lint 친화).
        pass

    def choose_action(self, state, legal_actions, env):
        del state, env
        import random
        weights = [9 - i for i in legal_actions]  # 0→9, 8→1
        return random.choices(legal_actions, weights=weights, k=1)[0], False

    def observe(self, *_args, **_kwargs):
        pass


def main() -> None:
    set_seed(SEED)
    results_dir = ensure_results_dir()
    env = TicTacToeEnv()

    # === Phase 1: vs RandomAgent (대칭 상대) ===
    print(f"[ex2] Phase 1: vs RandomAgent (num_games={NUM_GAMES})")
    set_seed(SEED)
    vanilla_vs_rand = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    history_vanilla_rand = train(
        vanilla_vs_rand, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    set_seed(SEED)
    sym_vs_rand = SymmetricTDAgent(alpha=ALPHA, epsilon=EPSILON)
    history_sym_rand = train(
        sym_vs_rand, RandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: RandomAgent(), eval_target="x",
    )

    # 상태 공간 크기 (H1.2a 정량적 효과)
    n_vanilla = vanilla_vs_rand.num_known_states()
    n_sym = sym_vs_rand.num_known_states()
    print(f"  알고있는 상태 수: vanilla={n_vanilla}, symmetric={n_sym} (비율 {n_vanilla/max(1,n_sym):.2f}x)")

    # H1.2a: 학습 곡선 비교
    plot_learning_curves(
        {"Vanilla TD": history_vanilla_rand, "Symmetric TD": history_sym_rand},
        "H1.2a/b: Learning curve vs RandomAgent (symmetric opponent)",
        results_dir / "ex2_h1ab_vs_random.png",
    )

    # 학습 초기 효율 확인 (앞쪽 30% 평균 win rate)
    early_n = max(1, len(history_vanilla_rand["eval_records"]) // 3)
    early_vanilla = sum(r["wins"] for r in history_vanilla_rand["eval_records"][:early_n]) / early_n
    early_sym = sum(r["wins"] for r in history_sym_rand["eval_records"][:early_n]) / early_n
    print(f"[H1.2a] 학습 초기({early_n}개 평가 평균) win rate: vanilla={early_vanilla:.3f}, sym={early_sym:.3f}")
    if early_sym > early_vanilla:
        print("  ✅ 대칭 활용이 초기 학습에서 우위 (H1.2a 지지)")
    else:
        print("  ⚠️ 초기 차이 미미 또는 역전 — 추가 분석 필요")

    final_vanilla = history_vanilla_rand["eval_records"][-1]
    final_sym = history_sym_rand["eval_records"][-1]
    print(f"[H1.2b] 최종 평가 (vs Random): vanilla={final_vanilla}, sym={final_sym}")
    if abs(final_vanilla["wins"] - final_sym["wins"]) < 0.05:
        print("  ✅ 대칭 상대에선 두 정책의 최종 성능 비슷 (H1.2b 지지)")
    else:
        print("  ⚠️ 큰 차이 — 가설 재검토")

    # === Phase 2: vs BiasedRandomAgent (비대칭 상대) ===
    print(f"\n[ex2] Phase 2: vs BiasedRandomAgent (비대칭 상대)")
    set_seed(SEED)
    vanilla_vs_biased = TDAgent(alpha=ALPHA, epsilon=EPSILON)
    history_vanilla_biased = train(
        vanilla_vs_biased, BiasedRandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: BiasedRandomAgent(), eval_target="x",
    )

    set_seed(SEED)
    sym_vs_biased = SymmetricTDAgent(alpha=ALPHA, epsilon=EPSILON)
    history_sym_biased = train(
        sym_vs_biased, BiasedRandomAgent(),
        num_games=NUM_GAMES, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES,
        eval_opponent_factory=lambda: BiasedRandomAgent(), eval_target="x",
    )

    plot_learning_curves(
        {"Vanilla TD": history_vanilla_biased, "Symmetric TD": history_sym_biased},
        "H1.2c: Learning curve vs BiasedRandomAgent (asymmetric opponent)",
        results_dir / "ex2_h1c_vs_biased.png",
    )

    final_van_biased = history_vanilla_biased["eval_records"][-1]
    final_sym_biased = history_sym_biased["eval_records"][-1]
    print(f"[H1.2c] 최종 평가 (vs BiasedRandom):")
    print(f"  vanilla TD: {final_van_biased}")
    print(f"  sym TD:     {final_sym_biased}")
    plot_outcome_distribution(
        {"Vanilla TD vs Biased": final_van_biased, "Symmetric TD vs Biased": final_sym_biased},
        "H1.2c: vs Asymmetric (BiasedRandom) opponent",
        results_dir / "ex2_h1c_vs_biased_bar.png",
    )
    if final_van_biased["wins"] > final_sym_biased["wins"]:
        print("  ✅ 비대칭 상대 활용에서 vanilla가 우위 (H1.2c 지지)")
    else:
        print("  ⚠️ 두 정책 성능 비슷 또는 역전 — BiasedRandom 정책을 충분히 활용 못 했을 수 있음")

    print(f"\n[ex2] 완료. 산출물: {results_dir}/ex2_*.png")


if __name__ == "__main__":
    main()
