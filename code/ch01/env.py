"""틱택토 환경.

OpenAI Gym 스타일 인터페이스로 구현.
상태(state)는 길이 9의 튜플로 표현 — hashable해서 dict 키로 사용 가능.

Reference: Sutton & Barto, RL: An Introduction (2nd Ed.), §1.5
"""

from __future__ import annotations  # 파이썬 3.9+ 에서 type hint 호환성

# === 상수 정의 ===
# 매직 넘버 금지 원칙: 의미 있는 이름으로
EMPTY = 0
PLAYER_X = 1   # 먼저 두는 쪽 (보통 학습 에이전트)
PLAYER_O = -1  # 나중 두는 쪽 (보통 상대)

BOARD_SIZE = 9  # 3x3

# 승리 라인 8개: 인덱스 0~8을 3x3 보드의
#   0 1 2
#   3 4 5
#   6 7 8
# 으로 매핑.
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 가로 3줄
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 세로 3줄
    (0, 4, 8), (2, 4, 6),             # 대각선 2개
]


# 타입 별칭: 상태는 9개 정수로 된 튜플
# 가독성을 위해 별칭 정의 (실제 동작에는 영향 없음)
State = tuple[int, ...]


class TicTacToeEnv:
    """틱택토 환경.

    사용 예:
        env = TicTacToeEnv()
        state = env.reset()
        while not done:
            action = agent.choose(state, env.legal_actions())
            state, reward, done, info = env.step(action)
    """

    def __init__(self):
        # __init__은 객체 생성 시 호출. 여기선 reset()을 부르기만.
        # 멤버 변수 초기화는 reset()에서 일괄 처리하면 코드 중복이 줄어듦.
        self.reset()

    def reset(self) -> State:
        """게임 초기화. 빈 보드 상태를 반환.

        Returns:
            빈 보드 상태 (모두 EMPTY).
        """
        # 보드는 내부적으로 list로 관리 (수정 가능해야 하니까)
        # 외부로 내보낼 때만 tuple로 변환 (불변성 보장)
        self._board = [EMPTY] * BOARD_SIZE
        self._current_player = PLAYER_X  # X가 먼저
        self._done = False
        self._winner = None  # 승자: PLAYER_X, PLAYER_O, 또는 None(무승부/진행중)
        return self._get_state()

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        """현재 플레이어가 action 위치에 두고, 결과를 반환.

        Args:
            action: 0~8 사이의 정수 (보드 위치).

        Returns:
            (next_state, reward, done, info)
            - next_state: 수를 둔 후의 상태
            - reward: 방금 둔 플레이어 관점에서의 보상
                      (승리:+1, 패배:-1, 무승부:0, 진행중:0)
            - done: 게임 종료 여부
            - info: 추가 정보 (디버깅용 dict)

        Raises:
            ValueError: 합법 수가 아닐 때.
        """
        # 방어적 코딩: 잘못된 입력은 명확한 에러로
        # "조용히 무시"하면 디버깅 지옥이 됨
        if self._done:
            raise ValueError("게임이 이미 종료되었습니다. reset()을 호출하세요.")
        if action not in self.legal_actions():
            raise ValueError(f"불법 수: {action}. 합법 수는 {self.legal_actions()}")

        # 둔 플레이어를 기억해두자 (보상 계산에 필요)
        player = self._current_player

        # 보드 갱신
        self._board[action] = player

        # 종료 체크
        reward = 0.0
        if self._check_win(player):
            self._done = True
            self._winner = player
            reward = 1.0  # 방금 둔 사람 승리
        elif self._is_board_full():
            self._done = True
            self._winner = None  # 무승부
            reward = 0.0
        else:
            # 게임 계속 → 턴 넘기기
            self._current_player = -self._current_player  # +1 ↔ -1 부호 반전

        info = {
            "winner": self._winner,
            "current_player": self._current_player,
        }
        return self._get_state(), reward, self._done, info

    def legal_actions(self) -> list[int]:
        """현재 둘 수 있는 위치 목록.

        Returns:
            빈 칸의 인덱스 리스트.
        """
        # list comprehension: 파이썬 관용 표현
        # 의미: "0~8 중에 빈 칸인 것만 모아라"
        return [i for i in range(BOARD_SIZE) if self._board[i] == EMPTY]

    @property
    def current_player(self) -> int:
        """현재 둘 차례인 플레이어. (PLAYER_X 또는 PLAYER_O)

        @property 데코레이터: 메서드를 속성처럼 호출 가능.
        env.current_player() 가 아니라 env.current_player 로 사용.
        """
        return self._current_player

    @property
    def winner(self) -> int | None:
        """게임 종료 시 승자. 무승부거나 진행 중이면 None."""
        return self._winner

    @property
    def done(self) -> bool:
        return self._done

    def simulate_step(self, action: int) -> State:
        """실제로 두지는 않고, action을 두면 갈 상태만 계산해서 반환.

        에이전트가 "어디 둘지" 결정하기 위해 각 후보 행동의 결과를
        미리 보는 용도 (afterstate value function 계산).

        Args:
            action: 시뮬레이션할 위치 (0~8).

        Returns:
            그 위치에 현재 플레이어가 둔 가상의 상태 (튜플).

        Raises:
            ValueError: 합법 수가 아니거나 게임이 종료된 경우.
        """
        if self._done:
            raise ValueError("게임이 종료되어 시뮬레이션 불가.")
        if action not in self.legal_actions():
            raise ValueError(f"불법 수 시뮬레이션: {action}")

        hypothetical = list(self._board)
        hypothetical[action] = self._current_player
        return tuple(hypothetical)

    def render(self) -> str:
        """보드를 사람이 읽기 좋은 문자열로 반환.

        디버깅용. print(env.render())로 사용.
        """
        symbols = {EMPTY: ".", PLAYER_X: "X", PLAYER_O: "O"}
        rows = []
        for r in range(3):
            row = " ".join(symbols[self._board[r * 3 + c]] for c in range(3))
            rows.append(row)
        return "\n".join(rows)

    # === 내부 헬퍼 메서드 (이름 앞 _는 "외부에서 직접 쓰지 마세요" 관례) ===

    def _get_state(self) -> State:
        """현재 보드를 hashable한 튜플로 반환."""
        return tuple(self._board)

    def _check_win(self, player: int) -> bool:
        """특정 플레이어가 승리 라인을 완성했는지 확인."""
        # any(): 하나라도 True면 True
        # 8개 승리 라인 중 하나라도 player로 가득 차면 승리
        return any(
            all(self._board[i] == player for i in line)
            for line in WIN_LINES
        )

    def _is_board_full(self) -> bool:
        """보드가 다 찼는지 확인."""
        return all(cell != EMPTY for cell in self._board)


# === 모듈을 직접 실행할 때만 동작하는 테스트 코드 ===
# 다른 파일에서 import할 때는 실행되지 않음
# `python env.py`로 실행하면 동작
if __name__ == "__main__":
    # 사람 vs 사람 콘솔 게임으로 환경 동작 확인
    print("=" * 40)
    print("틱택토 환경 테스트")
    print("=" * 40)
    print("위치 인덱스:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print()

    env = TicTacToeEnv()
    state = env.reset()

    while not env.done:
        print(env.render())
        symbol = "X" if env.current_player == PLAYER_X else "O"
        print(f"\n{symbol} 차례. 합법 수: {env.legal_actions()}")

        # 사용자 입력 받기
        try:
            action = int(input("위치 입력 (0-8): "))
            state, reward, done, info = env.step(action)
            print(f"보상: {reward}, 종료: {done}\n")
        except ValueError as e:
            print(f"에러: {e}\n")
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

    if env.done:
        print(env.render())
        if env.winner == PLAYER_X:
            print("\n🎉 X 승리!")
        elif env.winner == PLAYER_O:
            print("\n🎉 O 승리!")
        else:
            print("\n🤝 무승부!")