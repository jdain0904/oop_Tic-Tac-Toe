# =============================================================================
# environment.py — 4×4 틱택토 게임 환경
#
# 강화학습에서 "환경(Environment)"은 로봇이 행동하는 세계입니다.
# 로봇이 어떤 칸에 돌을 놓으면 환경이 다음 상태(board)와 보상(reward)을 돌려줍니다.
#
# 3×3 → 4×4 변경 포인트:
#   - 보드 크기: 9칸 → 16칸
#   - 승리 조건: 3줄 연속 → 4줄 연속
#   - 승리 패턴을 하드코딩 대신 BOARD_SIZE/WIN_LENGTH 에서 자동 계산합니다.
#     → config.py 의 숫자만 바꾸면 5×5, 6×6 등으로도 즉시 전환됩니다.
#
# 이 파일에 환경을 독립적으로 분리한 이유:
#   - 게임 규칙 변경이 이 파일만 수정하면 되도록 책임을 분리합니다.
#   - 여러 에이전트(O, X)가 같은 환경 객체를 공유해서 코드 중복을 없앱니다.
# =============================================================================

import numpy as np
import config


def _build_win_patterns(board_size: int, win_length: int) -> list[tuple]:
    """
    board_size × board_size 보드에서 win_length 개를 연속으로 놓으면 이기는
    모든 칸 조합(승리 패턴)을 자동으로 생성합니다.

    칸 번호 배치 예시 (4×4):
        0  |  1  |  2  |  3
        4  |  5  |  6  |  7
        8  |  9  | 10  | 11
       12  | 13  | 14  | 15

    생성되는 패턴 (4×4, win_length=4):
        가로 4개  × 1가지 시작위치 = 4패턴
        세로 4개  × 1가지 시작위치 = 4패턴
        대각선 ↘  1패턴
        대각선 ↗  1패턴
        합계: 10패턴
    """
    n = board_size
    w = win_length
    patterns = []

    # 가로: 각 행에서 w칸짜리 연속 구간
    for row in range(n):
        for col in range(n - w + 1):
            patterns.append(tuple(row * n + col + i for i in range(w)))

    # 세로: 각 열에서 w칸짜리 연속 구간
    for col in range(n):
        for row in range(n - w + 1):
            patterns.append(tuple((row + i) * n + col for i in range(w)))

    # 대각선 ↘ (왼쪽 위 → 오른쪽 아래)
    for row in range(n - w + 1):
        for col in range(n - w + 1):
            patterns.append(tuple((row + i) * n + col + i for i in range(w)))

    # 대각선 ↗ (왼쪽 아래 → 오른쪽 위)
    for row in range(w - 1, n):
        for col in range(n - w + 1):
            patterns.append(tuple((row - i) * n + col + i for i in range(w)))

    return patterns


class TicTacToeEnv:
    """
    N×N 틱택토 게임 환경 클래스. (현재 설정: 4×4, 4줄 승리)

    보드 표현:
        0  = 빈 칸
        1  = O 플레이어
       -1  = X 플레이어

    행동(action):
        0 ~ BOARD_SIZE²-1 사이의 정수 → 보드를 1차원으로 펼쳤을 때의 칸 번호
    """

    # 모듈 로드 시 한 번만 계산합니다. (매 게임마다 재계산하지 않아 속도 절약)
    WIN_PATTERNS = _build_win_patterns(config.BOARD_SIZE, config.WIN_LENGTH)
    N_CELLS = config.BOARD_SIZE * config.BOARD_SIZE

    def __init__(self):
        self.board = np.zeros(self.N_CELLS, dtype=int)
        self.current_player = 1   # O 선공
        self.done = False

    # ─────────────────────────────────────────────────────────────────────────
    # reset: 게임을 새로 시작합니다.
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self):
        """보드를 초기 상태로 되돌립니다. O가 항상 선공, X는 항상 후공."""
        self.board = np.zeros(self.N_CELLS, dtype=int)
        self.current_player = 1
        self.done = False
        return self._get_state()

    # ─────────────────────────────────────────────────────────────────────────
    # step: 현재 플레이어가 action 위치에 돌을 놓습니다.
    #
    # 반환값: (next_state, reward_o, reward_x, done)
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        보상 설계 원칙:
          승리  = +1.0 : 이기는 것이 최우선 목표
          패배  = -1.0 : 지는 것을 강하게 회피
          무승부 = +0.3 : 지는 것보다는 낫다는 명확한 신호
          진행  =  0.0 : 중간 보상 없이 최종 결과만으로 학습
                         (중간 보상을 주면 특정 위치를 맹목적으로 선호하는 편향 발생)
          잘못된 수 = -0.5 : 이미 채워진 칸 선택 시 패널티
                              패널티 없이 무시만 하면 규칙 학습이 지연됩니다.
        """
        # ── 잘못된 수 처리 ────────────────────────────────────────────────
        if self.board[action] != 0:
            if self.current_player == 1:
                return self._get_state(), config.REWARD_INVALID, config.REWARD_STEP, False
            else:
                return self._get_state(), config.REWARD_STEP, config.REWARD_INVALID, False

        # ── 돌 놓기 ───────────────────────────────────────────────────────
        self.board[action] = self.current_player

        # ── 승리 판정 ─────────────────────────────────────────────────────
        if self._check_winner(self.current_player):
            self.done = True
            if self.current_player == 1:
                return self._get_state(), config.REWARD_WIN, config.REWARD_LOSE, True
            else:
                return self._get_state(), config.REWARD_LOSE, config.REWARD_WIN, True

        # ── 무승부 판정 ───────────────────────────────────────────────────
        if not self._has_empty():
            self.done = True
            return self._get_state(), config.REWARD_DRAW, config.REWARD_DRAW, True

        # ── 게임 계속 — 플레이어 교체 ────────────────────────────────────
        self.current_player *= -1
        return self._get_state(), config.REWARD_STEP, config.REWARD_STEP, False

    # ─────────────────────────────────────────────────────────────────────────
    # get_valid_actions: 현재 보드에서 둘 수 있는 칸 목록을 반환합니다.
    # ─────────────────────────────────────────────────────────────────────────
    def get_valid_actions(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    # ─────────────────────────────────────────────────────────────────────────
    # render: 현재 보드를 터미널에 출력합니다.
    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        """4×4 보드를 칸 번호와 함께 출력합니다."""
        symbols = {0: ".", 1: "O", -1: "X"}
        n = config.BOARD_SIZE
        sep = "  " + "+".join(["----"] * n)
        print()
        for row in range(n):
            cells = [f" {symbols[self.board[row * n + col]]:^2}" for col in range(n)]
            print("  " + "|".join(cells))
            if row < n - 1:
                print(sep)
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> tuple:
        """보드를 Q-테이블 키로 쓸 수 있도록 tuple 로 변환합니다."""
        return tuple(self.board)

    def _check_winner(self, player: int) -> bool:
        """player 가 WIN_PATTERNS 중 하나를 완성했는지 확인합니다."""
        for pattern in self.WIN_PATTERNS:
            if all(self.board[i] == player for i in pattern):
                return True
        return False

    def _has_empty(self) -> bool:
        return 0 in self.board
