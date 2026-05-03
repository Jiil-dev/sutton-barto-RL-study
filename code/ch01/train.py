"""학습 루프 (env + agent 연결).

핵심 함수:
    play_game(env, agent_x, agent_o, learn=True) -> int | None
        한 게임을 끝까지 진행하고 승자를 반환. 학습 모드면 양쪽 모두 observe() 호출.

    train(agent_x, agent_o, num_games, eval_every=..., eval_opponent_factory=..., ...)
        num_games번 학습. eval_every마다 평가하여 학습 곡선용 history 반환.

    evaluate(agent, opponent_factory, num_games, agent_plays_x=...)
        agent의 training/exploration을 끄고 opponent와 대국. (wins, losses, draws) 반환.

설계 메모:
- 한 게임에 항상 두 에이전트 인스턴스(agent_x, agent_o)가 필요. 자가 대국이라도
  같은 인스턴스를 두 번 넘기면 values dict가 양쪽 시점으로 오염되므로 금지.
  (자가 대국은 동일 클래스의 **두 인스턴스**를 사용해야 함. ex1 스크립트에서 처리.)
- observe()는 "그 수를 둔 에이전트"에게만 호출. 게임 종료 시점에는 마지막 수를 두지
  않은 쪽(패자 또는 무승부 측)에게 별도로 terminal observe를 호출하여 부정 보상이
  학습에 반영되게 함.
- 평가는 reproducibility를 위해 양쪽의 training 토글을 잠시 False로 강제하고 복원.

Reference: Sutton & Barto, RL: An Introduction (2nd Ed.), §1.5
"""

from __future__ import annotations

import random
from typing import Callable, Protocol

from env import EMPTY, PLAYER_O, PLAYER_X, TicTacToeEnv

State = tuple[int, ...]


class Agent(Protocol):
    """학습 루프가 기대하는 에이전트 프로토콜.

    TDAgent / RandomAgent / QAgent 모두 만족해야 함.
    실제 isinstance 체크는 안 하지만 IDE/타입 체커용 문서 역할.
    """

    training: bool

    def choose_action(
        self, state: State, legal_actions: list[int], env: TicTacToeEnv
    ) -> tuple[int, bool]: ...

    def observe(
        self,
        prev_afterstate: State | None,
        curr_afterstate: State,
        reward: float,
        done: bool,
        was_exploratory: bool,
    ) -> None: ...


# === 한 게임 진행 ===

def play_game(
    env: TicTacToeEnv,
    agent_x: Agent,
    agent_o: Agent,
    learn: bool = True,
) -> int | None:
    """한 게임을 끝까지 진행하고 승자를 반환.

    Args:
        env: 틱택토 환경. 함수 내부에서 reset() 호출.
        agent_x: PLAYER_X 측 에이전트.
        agent_o: PLAYER_O 측 에이전트.
        learn: True면 양쪽 모두 observe() 호출. False면 평가용 (학습 신호 없음).

    Returns:
        env.winner: PLAYER_X(+1), PLAYER_O(-1), 또는 None(무승부).
    """
    agents: dict[int, Agent] = {PLAYER_X: agent_x, PLAYER_O: agent_o}

    state = env.reset()
    # 각 에이전트의 "직전에 본인이 둔 후의 afterstate"와 그때의 was_exploratory를 추적.
    # TD 갱신은 "두 직후의 afterstate" 단위로 일어나는데, 한 게임에서 X와 O가 번갈아
    # 두므로 각자의 마지막 afterstate를 따로 기억해야 한다.
    last_afterstate: dict[int, State | None] = {PLAYER_X: None, PLAYER_O: None}
    last_was_exp: dict[int, bool] = {PLAYER_X: False, PLAYER_O: False}

    while not env.done:
        player = env.current_player
        agent = agents[player]
        legal = env.legal_actions()

        action, was_exp = agent.choose_action(state, legal, env)

        # 두기 직전에 simulate로 afterstate(=내가 둔 직후의 보드) 미리 계산.
        # env.step()을 호출하면 current_player가 토글되어 다음 simulate가 의미를 잃기에
        # 호출 전에 잡아둬야 함.
        afterstate = env.simulate_step(action)
        state, reward, done, _ = env.step(action)

        if learn:
            # 본인 시점의 TD 갱신: prev_afterstate(직전 본인 수) → afterstate(이번 본인 수).
            agent.observe(last_afterstate[player], afterstate, reward, done, was_exp)
        last_afterstate[player] = afterstate
        last_was_exp[player] = was_exp

    # === 게임 종료 후 정리: 마지막에 두지 않은 쪽 ===
    # env.current_player는 마지막에 둔 사람으로 남아있다 (env.step에서 종료 시 토글 안 함).
    # 마지막에 두지 않은 사람(other)은 자신의 마지막 afterstate가 결과적으로
    # 패배(또는 무승부)로 이어졌음을 학습해야 한다.
    if learn:
        last_mover = env.current_player
        other = -last_mover
        if last_afterstate[other] is not None:
            # 무승부면 0, 아니면 마지막에 둔 쪽이 이긴 것이므로 other 입장에선 -1.
            other_reward = 0.0 if env.winner is None else -1.0
            # curr_afterstate로 종료 보드(state)를 그대로 넘긴다. observe() 내부의
            # `if done: values[curr_afterstate] = reward` 로직에 의해 other의 values dict에는
            # "이 종료 보드 = other_reward"로 저장되고, 그 값이 마지막 afterstate를 backup하는
            # 타깃으로 사용된다. (각 에이전트가 독립된 values dict를 가지므로 다른 에이전트의
            # 종료 보드 값과 충돌하지 않는다.)
            agents[other].observe(
                last_afterstate[other],
                state,
                other_reward,
                True,
                last_was_exp[other],
            )

    return env.winner


# === 평가 ===

def evaluate(
    agent: Agent,
    opponent_factory: Callable[[], Agent],
    num_games: int,
    agent_plays_x: bool = True,
    env: TicTacToeEnv | None = None,
) -> dict[str, float]:
    """학습된 agent를 frozen 모드로 평가.

    Args:
        agent: 평가 대상 에이전트.
        opponent_factory: 매 게임마다 새 상대 인스턴스를 만드는 함수.
            (예: `lambda: RandomAgent()`). 매번 새로 만드는 이유는 상대도 학습형이라면
            평가 도중 학습되지 않도록 격리하기 위함. 무상태 에이전트라면 같은 인스턴스를
            돌려줘도 무방.
        num_games: 평가 게임 수.
        agent_plays_x: True면 agent가 X(선공), False면 O(후공).
        env: 재사용할 환경. None이면 새로 만듦.

    Returns:
        {"wins": float, "losses": float, "draws": float} 비율 (합 = 1.0).
    """
    if env is None:
        env = TicTacToeEnv()

    # agent의 training 플래그를 잠시 False로 (탐험 OFF, 학습 OFF).
    # 평가가 끝나면 원래 값으로 복원해서 학습 루프에 부작용을 남기지 않는다.
    original_training = agent.training
    agent.training = False

    wins = losses = draws = 0
    try:
        for _ in range(num_games):
            opponent = opponent_factory()
            opponent.training = False  # 상대도 평가 모드

            if agent_plays_x:
                winner = play_game(env, agent, opponent, learn=False)
                agent_side = PLAYER_X
            else:
                winner = play_game(env, opponent, agent, learn=False)
                agent_side = PLAYER_O

            if winner == agent_side:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1
    finally:
        agent.training = original_training

    return {
        "wins": wins / num_games,
        "losses": losses / num_games,
        "draws": draws / num_games,
    }


# === 학습 루프 ===

def train(
    agent_x: Agent,
    agent_o: Agent,
    num_games: int,
    eval_every: int | None = None,
    eval_games: int = 200,
    eval_opponent_factory: Callable[[], Agent] | None = None,
    eval_target: str = "x",  # "x" or "o" — 어느 쪽을 평가 대상으로 삼을지
    show_progress: bool = False,
) -> dict:
    """num_games번 학습. eval_every마다 학습 곡선용 평가 기록.

    Args:
        agent_x, agent_o: 학습할 에이전트들. 자가 대국이면 같은 클래스의 **별도 인스턴스**.
        num_games: 학습 게임 수.
        eval_every: N게임마다 평가 (None이면 평가 없음).
        eval_games: 평가 1회당 게임 수.
        eval_opponent_factory: 평가 시 상대 (보통 무작위 베이스라인).
            None이면 평가 비활성화.
        eval_target: "x" 또는 "o" — eval 시 어느 에이전트를 평가할지.
        show_progress: tqdm 진행바 표시 여부.

    Returns:
        history: {
            "results": [winner per game, ...],   # 길이 num_games
            "eval_steps": [step, ...],            # 평가 시점 (게임 인덱스)
            "eval_records": [evaluate() 결과 dict, ...],  # 같은 길이
        }
    """
    if show_progress:
        # tqdm은 옵션 의존성. 호출 시점에만 import하여 환경 의존성 줄임.
        from tqdm import tqdm
        iterator = tqdm(range(num_games), desc="train")
    else:
        iterator = range(num_games)

    env = TicTacToeEnv()
    eval_env = TicTacToeEnv()  # 평가용 별도 환경 (학습 환경과 상태 섞임 방지)

    history: dict = {"results": [], "eval_steps": [], "eval_records": []}
    target = agent_x if eval_target == "x" else agent_o

    for i in iterator:
        winner = play_game(env, agent_x, agent_o, learn=True)
        history["results"].append(winner)

        # 평가 트리거: eval_every가 양수일 때만, 그리고 매 eval_every번째에.
        # i+1을 쓰는 이유: 0번째 게임 직후 평가하면 학습 전 결과라 의미 없음.
        if eval_every and eval_opponent_factory is not None and (i + 1) % eval_every == 0:
            record = evaluate(
                target,
                eval_opponent_factory,
                num_games=eval_games,
                agent_plays_x=(eval_target == "x"),
                env=eval_env,
            )
            history["eval_steps"].append(i + 1)
            history["eval_records"].append(record)

    return history


# === 자가 테스트 ===
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # agents/ 패키지 import를 위해 code/ch01/을 path에 추가
    # (이 파일이 직접 실행되는 경우의 import 보정).
    sys.path.insert(0, str(Path(__file__).parent))

    from agents.random_agent import RandomAgent
    from agents.td_agent import TDAgent

    print("=" * 50)
    print("train.py 자가 테스트")
    print("=" * 50)

    random.seed(42)

    # === 테스트 1: play_game이 정상 종료 (Random vs Random) ===
    env = TicTacToeEnv()
    a, b = RandomAgent(), RandomAgent()
    winner = play_game(env, a, b, learn=False)
    assert env.done, "게임이 종료 상태여야 함"
    assert winner in (PLAYER_X, PLAYER_O, None), f"잘못된 승자: {winner}"
    print(f"[1] Random vs Random 한 판: winner={winner}")

    # === 테스트 2: TD가 학습으로 상태를 본다 (값 dict이 자라야 함) ===
    td_x = TDAgent(alpha=0.3, epsilon=0.2)
    rand_o = RandomAgent()
    before = td_x.num_known_states()
    for _ in range(50):
        play_game(env, td_x, rand_o, learn=True)
    after = td_x.num_known_states()
    assert after > before, f"학습 후 상태가 늘어야 함: {before} -> {after}"
    print(f"[2] TD vs Random 50판 후 알고있는 상태: {before} -> {after}")

    # === 테스트 3: evaluate가 합이 1인 비율을 반환 ===
    rec = evaluate(td_x, lambda: RandomAgent(), num_games=100)
    s = rec["wins"] + rec["losses"] + rec["draws"]
    assert abs(s - 1.0) < 1e-9, f"승/패/무 합이 1이 아님: {s}"
    print(f"[3] evaluate (50판 학습 후 vs Random 100판): {rec}")

    # === 테스트 4: evaluate가 agent.training 플래그를 복원 ===
    td_x.training = True
    evaluate(td_x, lambda: RandomAgent(), num_games=10)
    assert td_x.training is True, "evaluate 후 training 플래그가 복원되어야 함"
    td_x.training = False
    evaluate(td_x, lambda: RandomAgent(), num_games=10)
    assert td_x.training is False, "evaluate 후 training=False도 유지되어야 함"
    print("[4] evaluate가 training 플래그를 정확히 복원")

    # === 테스트 5: train()이 충분 학습 후 무작위 상대를 압도 ===
    td_x2 = TDAgent(alpha=0.2, epsilon=0.1)
    rand_o2 = RandomAgent()
    history = train(
        td_x2, rand_o2,
        num_games=2000,
        eval_every=500,
        eval_games=200,
        eval_opponent_factory=lambda: RandomAgent(),
        eval_target="x",
    )
    final_eval = history["eval_records"][-1]
    print(f"[5] TD 2000판 학습 후 평가: {final_eval}")
    assert final_eval["wins"] >= 0.7, (
        f"2000판 학습 후 승률이 너무 낮음 (>=0.7 기대): {final_eval}"
    )
    assert len(history["results"]) == 2000
    assert len(history["eval_records"]) == 4  # 500, 1000, 1500, 2000

    # === 테스트 6: 자가 대국 (별도 인스턴스 두 개) ===
    td_a = TDAgent(alpha=0.1, epsilon=0.1)
    td_b = TDAgent(alpha=0.1, epsilon=0.1)
    for _ in range(200):
        play_game(env, td_a, td_b, learn=True)
    assert td_a.num_known_states() > 0 and td_b.num_known_states() > 0
    print(
        f"[6] 자가 대국 200판 후 (X상태수={td_a.num_known_states()}, "
        f"O상태수={td_b.num_known_states()})"
    )

    # === 테스트 7: 종료 시 패자 측에도 V[terminal] = -1이 저장됐는지 ===
    # 200판 자가 대국 중 td_b가 진 적이 한 번이라도 있다면 그 종료보드 V는 -1로 저장.
    # 정확한 보드를 잡기 어렵지만, 음수 V가 적어도 하나 있는지로 sanity 체크.
    has_negative_b = any(v < 0 for v in td_b.values.values())
    assert has_negative_b, "패배 시 terminal V=-1 저장이 안 된 듯"
    print("[7] 패자측 terminal V=-1 저장 sanity 통과")

    print("\n✅ 모든 테스트 통과")
