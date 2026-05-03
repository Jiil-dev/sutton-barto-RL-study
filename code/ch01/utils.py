"""실험 공통 헬퍼: 시각화, 저장/로드, 재현성, 정책 분석.

설계 메모:
- matplotlib는 `Agg` 백엔드로 강제. SSH/헤드리스 환경에서도 PNG 저장만 잘 되면 OK.
- 가치 dict 저장은 `pickle` (tuple 키를 그대로 직렬화). JSON은 tuple 키를 못 받음.
- random 시드는 random 모듈에만 적용. 다른 라이브러리(numpy 등)는 현재 학습 경로에 없으므로
  필요 시 그때 추가 (YAGNI).
- 8개 대칭(D4) 변환 매핑은 ex2 + 일반 분석에 둘 다 쓰일 수 있어 여기에 둠.
"""

from __future__ import annotations

import pickle
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import matplotlib

# PNG로만 저장하므로 헤드리스 백엔드 강제 (디스플레이 없는 환경에서 import 에러 방지).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from env import EMPTY, PLAYER_O, PLAYER_X, BOARD_SIZE  # noqa: E402

if TYPE_CHECKING:
    from env import TicTacToeEnv

State = tuple[int, ...]


# === 결과 저장 디렉토리 ===
# 프로젝트 루트(=code/ch01의 두 단계 위)/results/ch01 로 통일.
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ch01"


def ensure_results_dir() -> Path:
    """결과 디렉토리를 생성하고 경로 반환."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# === 재현성 ===

def set_seed(seed: int) -> None:
    """전역 random 시드 고정.

    에이전트 내부에 별도 RNG가 없으므로 한 번 호출이면 모든 무작위 분기가 결정론적이 됨.
    """
    random.seed(seed)


# === 저장/로드 ===

def save_values(values: dict[State, float], path: str | Path) -> None:
    """학습된 가치 함수(dict) 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(values, f)


def load_values(path: str | Path) -> dict[State, float]:
    """저장된 가치 함수 로드."""
    with open(path, "rb") as f:
        return pickle.load(f)


# === 학습 곡선 시각화 ===

def plot_learning_curves(
    histories: dict[str, dict],
    title: str,
    save_path: str | Path,
    metric: str = "wins",
) -> None:
    """여러 학습 기록을 한 그래프로 비교.

    Args:
        histories: {레이블: history dict (train()이 반환한 것)} 매핑.
        title: 그래프 제목.
        save_path: PNG 저장 경로.
        metric: "wins" | "losses" | "draws" 중 하나.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        steps = history["eval_steps"]
        ys = [r[metric] for r in history["eval_records"]]
        ax.plot(steps, ys, marker="o", label=label, linewidth=2)
    ax.set_xlabel("Training games")
    ax.set_ylabel(f"Eval {metric} rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_outcome_distribution(
    outcomes: dict[str, dict[str, float]],
    title: str,
    save_path: str | Path,
) -> None:
    """막대그래프: 정책별 win/loss/draw 비율 비교.

    Args:
        outcomes: {레이블: {"wins": x, "losses": y, "draws": z}} 매핑.
        title, save_path: 자명.
    """
    labels = list(outcomes.keys())
    x = range(len(labels))
    wins = [outcomes[k]["wins"] for k in labels]
    losses = [outcomes[k]["losses"] for k in labels]
    draws = [outcomes[k]["draws"] for k in labels]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5))
    width = 0.25
    ax.bar([i - width for i in x], wins, width, label="wins", color="#2a9d8f")
    ax.bar(list(x), draws, width, label="draws", color="#e9c46a")
    ax.bar([i + width for i in x], losses, width, label="losses", color="#e76f51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_first_move_heatmap(
    distribution: dict[int, float],
    title: str,
    save_path: str | Path,
) -> None:
    """첫 수 분포를 3x3 히트맵으로 시각화."""
    grid = [[0.0] * 3 for _ in range(3)]
    for pos, p in distribution.items():
        grid[pos // 3][pos % 3] = p

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=max(0.01, max(distribution.values())))
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{grid[r][c]:.2f}", ha="center", va="center",
                    color="white" if grid[r][c] < 0.5 else "black", fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# === 정책 분석 ===

def first_move_distribution(
    agent,
    env: "TicTacToeEnv",
    n_samples: int = 1000,
) -> dict[int, float]:
    """빈 보드에서 agent의 첫 수 분포 (위치별 선택 비율).

    훈련된 정책이 어느 위치를 선호하는지 확인용. 평가 모드(training=False)로 측정해
    탐험 노이즈를 제거.
    """
    state = env.reset()
    legal = env.legal_actions()
    counts: Counter = Counter()

    original_training = agent.training
    agent.training = False
    try:
        for _ in range(n_samples):
            # 매번 같은 상태에서 호출. tie-breaking은 random이라 분포가 나옴.
            action, _ = agent.choose_action(state, legal, env)
            counts[action] += 1
    finally:
        agent.training = original_training

    return {pos: counts.get(pos, 0) / n_samples for pos in range(BOARD_SIZE)}


def average_game_length(
    play_fn,  # 함수: () -> list of moves (from env perspective)
    n_games: int = 100,
) -> float:
    """평균 게임 길이 측정. 호출 측에서 한 게임당 수 횟수 리스트를 yield하는 헬퍼 사용."""
    lengths = list(play_fn() for _ in range(n_games))
    return sum(lengths) / len(lengths)


def measure_game_length(
    env: "TicTacToeEnv",
    agent_x,
    agent_o,
    n_games: int = 100,
) -> float:
    """평균 게임 길이 (수 횟수). play_game을 직접 다시 짠 이유:
    play_game은 winner만 반환하고 게임 길이는 env._board의 비-EMPTY 칸 수로 사후 추정 가능.
    """
    total = 0
    for _ in range(n_games):
        env.reset()
        # 학습 신호 없이 play_game을 한 번 돌리고, 종료 후 채워진 칸 수를 셈.
        # 학습 모드 비활성화 보장.
        original_x = agent_x.training
        original_o = agent_o.training
        agent_x.training = False
        agent_o.training = False
        try:
            from train import play_game  # 순환 import 방지: 함수 내부에서.
            play_game(env, agent_x, agent_o, learn=False)
            # env._board에 접근하지 않고 공개 인터페이스로만 길이 측정:
            # 종료 보드를 받기 위해 임시로 reset() 전 직접 계산이 어렵다.
            # 대신 play_game 직후 env.legal_actions()의 보수(9 - 합법 수)로 채워진 칸 수.
            filled = BOARD_SIZE - len(env.legal_actions())
            total += filled
        finally:
            agent_x.training = original_x
            agent_o.training = original_o
    return total / n_games


# === 대칭성 (D4 group, 8개) ===
# 각 변환은 length-9 보드 상의 인덱스 순열로 표현.
# 보드 인덱스:
#   0 1 2
#   3 4 5
#   6 7 8

def _rotate90(perm: list[int]) -> list[int]:
    """주어진 순열 위에 추가로 90도 회전을 합성한 결과를 반환."""
    # 90도 회전 매핑: (r,c) -> (c, 2-r). new[c*3 + (2-r)] = old[r*3 + c]
    rot = [0] * 9
    for r in range(3):
        for c in range(3):
            rot[c * 3 + (2 - r)] = perm[r * 3 + c]
    return rot


def _flip_horizontal(perm: list[int]) -> list[int]:
    """좌우 반사 합성."""
    flip = [0] * 9
    for r in range(3):
        for c in range(3):
            flip[r * 3 + (2 - c)] = perm[r * 3 + c]
    return flip


def _generate_symmetries() -> list[list[int]]:
    """D4 8개 변환을 보드 인덱스 순열로 반환. 첫 번째는 항등."""
    identity = list(range(9))
    perms = [identity]
    cur = identity
    for _ in range(3):  # 90, 180, 270도 회전
        cur = _rotate90(cur)
        perms.append(cur)
    # 4개 회전 각각에 대해 좌우 반사 추가
    flipped = [_flip_horizontal(p) for p in perms]
    perms.extend(flipped)
    return perms


SYMMETRIES: list[list[int]] = _generate_symmetries()


def apply_symmetry(state: State, perm: list[int]) -> State:
    """state에 인덱스 순열을 적용한 변환 결과 반환."""
    # perm[i] = j 의미: 원본의 j번 칸이 변환 후 i번 자리로 감.
    # → new[i] = state[perm_inverse[i]] 형태로 풀어야 하지만, 우리는 _rotate90/_flip을
    # "위치 매핑"으로 정의했으므로 직접 적용:
    # _rotate90 정의를 따라 perm[i]는 변환 후 i번 자리에 들어갈 원본 인덱스.
    # 그러나 _rotate90의 코드를 다시 보면 rot[c*3+(2-r)] = perm[r*3+c]:
    # 이는 "원본 위치 (r,c)의 값을 새 위치 (c, 2-r)에 둔다"는 의미라
    # 즉 perm을 "변환 함수처럼 각 위치 i가 어디로 가는지"가 아니라
    # "이 변환을 항등에서 시작해 누적 적용했을 때 각 새 위치에 원본 어느 인덱스 값이 오는지"를
    # 표현한다. 따라서 new_state[i] = state[perm[i]] 가 맞음.
    return tuple(state[perm[i]] for i in range(BOARD_SIZE))


def canonical_state(state: State) -> State:
    """8개 대칭 중 사전식 최소 형태를 정규형으로 반환."""
    return min(apply_symmetry(state, p) for p in SYMMETRIES)


# === 자가 테스트 ===

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from agents.random_agent import RandomAgent
    from agents.td_agent import TDAgent
    from env import TicTacToeEnv

    print("=" * 50)
    print("utils.py 자가 테스트")
    print("=" * 50)

    set_seed(0)

    # 1) 결과 디렉토리 생성
    rd = ensure_results_dir()
    assert rd.exists()
    print(f"[1] results dir: {rd}")

    # 2) save_values / load_values 라운드트립
    sample_values: dict[State, float] = {
        (0, 0, 0, 0, 0, 0, 0, 0, 0): 0.5,
        (1, 0, 0, 0, 0, 0, 0, 0, 0): 0.7,
    }
    tmp_path = rd / "_test_values.pkl"
    save_values(sample_values, tmp_path)
    loaded = load_values(tmp_path)
    assert loaded == sample_values, "저장/로드 라운드트립 실패"
    tmp_path.unlink()  # 정리
    print("[2] save/load 라운드트립 OK")

    # 3) 대칭 8개 정확성: 빈 보드는 모든 대칭 하에서 빈 보드
    empty = (0,) * 9
    for p in SYMMETRIES:
        assert apply_symmetry(empty, p) == empty
    # 모서리/변/중앙 한 곳씩 두면 대칭으로 묶이는 종류가 정확해야 함
    # 모서리 4개, 변 4개, 중앙 1개 → 한 자리에 1을 두고 8개 대칭 적용 시
    # 모서리 단일 배치는 4개 distinct 보드, 변 단일 배치는 4개 distinct 보드, 중앙은 1개.
    def distinct_after_sym(idx: int) -> int:
        s = [0] * 9
        s[idx] = 1
        s = tuple(s)
        return len({apply_symmetry(s, p) for p in SYMMETRIES})

    assert distinct_after_sym(0) == 4, "모서리(0)의 대칭 클래스 크기"
    assert distinct_after_sym(1) == 4, "변(1)의 대칭 클래스 크기"
    assert distinct_after_sym(4) == 1, "중앙(4)의 대칭 클래스 크기"
    print("[3] D4 대칭 8개 정확성 통과")

    # 4) canonical_state는 같은 대칭 클래스에 대해 동일 결과
    s_corner1 = (1,) + (0,) * 8
    s_corner2 = (0, 0, 1) + (0,) * 6
    assert canonical_state(s_corner1) == canonical_state(s_corner2)
    print("[4] canonical_state: 같은 대칭 클래스는 같은 정규형")

    # 5) first_move_distribution: 빈 보드 + 학습 안 한 TDAgent → 9칸이 거의 균등
    env = TicTacToeEnv()
    agent = TDAgent()
    dist = first_move_distribution(agent, env, n_samples=900)
    assert len(dist) == 9
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # 학습 전이라 모든 V=0이므로 tie-breaking에 의해 균등 분포 (각 ~0.111)
    for pos, p in dist.items():
        assert 0.05 < p < 0.20, f"위치 {pos} 분포 비정상: {p}"
    print(f"[5] first_move_distribution (학습 전 TD): {dist}")

    # 6) measure_game_length: Random vs Random 평균 게임 길이는 5~9 사이
    env2 = TicTacToeEnv()
    avg = measure_game_length(env2, RandomAgent(), RandomAgent(), n_games=50)
    assert 5 <= avg <= 9, f"평균 게임 길이 범위 비정상: {avg}"
    print(f"[6] 평균 게임 길이 (Random vs Random, 50판): {avg:.2f}")

    # 7) plot 함수: 에러 없이 PNG 생성하는지
    fake_history = {
        "test1": {
            "results": [],
            "eval_steps": [10, 20, 30],
            "eval_records": [
                {"wins": 0.5, "losses": 0.3, "draws": 0.2},
                {"wins": 0.7, "losses": 0.2, "draws": 0.1},
                {"wins": 0.85, "losses": 0.05, "draws": 0.10},
            ],
        }
    }
    plot_learning_curves(fake_history, "test", rd / "_test_curve.png")
    plot_outcome_distribution(
        {"a": {"wins": 0.5, "losses": 0.3, "draws": 0.2}}, "test",
        rd / "_test_bar.png",
    )
    plot_first_move_heatmap(dist, "test", rd / "_test_heat.png")
    for fname in ["_test_curve.png", "_test_bar.png", "_test_heat.png"]:
        p = rd / fname
        assert p.exists() and p.stat().st_size > 0
        p.unlink()
    print("[7] plot 함수 모두 PNG 생성 OK (테스트 파일 정리 완료)")

    print("\n✅ 모든 테스트 통과")
