# =============================================================================
# play.py — 학습된 에이전트 평가 및 사람 vs 로봇 대전
#
# 이 파일에서 할 수 있는 것:
#   1. evaluate() — 두 로봇을 1,000판 대결시켜 성능을 수치로 확인
#   2. play_vs_human() — 사람이 직접 O 또는 X 로봇과 대전
#
# 이 파일을 분리한 이유:
#   - 학습(train.py)과 평가/플레이(play.py)는 목적이 다릅니다.
#     학습 중에는 화면 출력을 최소화하고, 평가 시에는 보드를 보여줘야 합니다.
#   - train.py 를 수정하지 않고도 평가 방식을 자유롭게 바꿀 수 있습니다.
# =============================================================================

import config
from environment import TicTacToeEnv
from agent import QLearningAgent


# =============================================================================
# 1. 에이전트 성능 평가
# =============================================================================

def evaluate(n_games: int = 1000):
    """
    학습된 O 로봇과 X 로봇을 n_games 판 대결시켜 최종 승률을 출력합니다.
    평가 중에는 탐험(무작위 수)을 완전히 끄고 최선의 수만 사용합니다.
    """
    print("=" * 60)
    print(f"  성능 평가: {n_games:,}판 대결 (탐험 없음)")
    print("=" * 60)

    env = TicTacToeEnv()
    agent_o = QLearningAgent("O")
    agent_x = QLearningAgent("X")

    agent_o.load(config.MODEL_PATH_O)
    agent_x.load(config.MODEL_PATH_X)

    # 평가 시에는 탐험을 끕니다 — 배운 것을 100% 활용하여 실력을 측정합니다.
    agent_o.epsilon = 0.0
    agent_x.epsilon = 0.0

    results = {"O_win": 0, "X_win": 0, "draw": 0}

    for _ in range(n_games):
        state = env.reset()
        while not env.done:
            valid = env.get_valid_actions()
            if env.current_player == 1:
                action = agent_o.choose_action(state, valid)
            else:
                action = agent_x.choose_action(state, valid)
            state, r_o, r_x, done = env.step(action)

        if r_o == config.REWARD_WIN:
            results["O_win"] += 1
        elif r_x == config.REWARD_WIN:
            results["X_win"] += 1
        else:
            results["draw"] += 1

    print(f"O 로봇 승: {results['O_win']:>5} ({results['O_win']/n_games*100:.1f}%)")
    print(f"X 로봇 승: {results['X_win']:>5} ({results['X_win']/n_games*100:.1f}%)")
    print(f"무 승 부:  {results['draw']:>5} ({results['draw']/n_games*100:.1f}%)")
    print()

    # 완벽한 틱택토 전략에서는 양쪽이 최선을 다하면 항상 무승부입니다.
    # 무승부 비율이 높을수록 두 로봇 모두 잘 학습된 것입니다.
    draw_rate = results["draw"] / n_games * 100
    if draw_rate >= 90:
        print("✓ 두 로봇 모두 최적 전략에 매우 가깝게 학습되었습니다.")
    elif draw_rate >= 70:
        print("△ 어느 정도 학습되었지만 아직 개선 여지가 있습니다.")
    else:
        print("✗ 학습이 더 필요합니다. train.py 를 다시 실행해 보세요.")


# =============================================================================
# 2. 사람 vs 로봇 대전
# =============================================================================

def play_vs_human():
    """
    사람이 터미널에서 로봇과 직접 대전합니다.

    칸 번호 안내 (4×4):
         0 |  1 |  2 |  3
        ----+----+----+----
         4 |  5 |  6 |  7
        ----+----+----+----
         8 |  9 | 10 | 11
        ----+----+----+----
        12 | 13 | 14 | 15
    """
    print("=" * 60)
    print("  사람 vs 로봇 대전")
    print("=" * 60)

    # 어느 로봇과 대전할지 선택
    side = input("O 로봇(1) 또는 X 로봇(2) 중 선택하세요 [1/2]: ").strip()
    robot_is_o = (side != "2")

    env = TicTacToeEnv()
    agent_o = QLearningAgent("O")
    agent_x = QLearningAgent("X")
    agent_o.load(config.MODEL_PATH_O)
    agent_x.load(config.MODEL_PATH_X)

    # 평가 모드: 탐험 없이 최선의 수만 사용합니다.
    agent_o.epsilon = 0.0
    agent_x.epsilon = 0.0

    human_symbol = "X" if robot_is_o else "O"
    robot_symbol = "O" if robot_is_o else "X"
    n_cells = config.BOARD_SIZE * config.BOARD_SIZE
    print(f"\n당신은 [{human_symbol}] 로봇은 [{robot_symbol}]  (O가 먼저 시작)")
    print(f"칸 번호: 0~{n_cells - 1} 을 입력하세요.\n")

    state = env.reset()

    while not env.done:
        env.render()
        valid = env.get_valid_actions()

        # 현재 차례 결정
        is_robot_turn = (env.current_player == 1 and robot_is_o) or \
                        (env.current_player == -1 and not robot_is_o)

        if is_robot_turn:
            # 로봇 차례
            agent = agent_o if env.current_player == 1 else agent_x
            action = agent.choose_action(state, valid)
            print(f"  로봇 [{robot_symbol}] 이 칸 {action} 에 두었습니다.")
        else:
            # 사람 차례
            while True:
                try:
                    action = int(input(f"  당신 [{human_symbol}] 차례 — 칸 번호 입력 (유효: {valid}): "))
                    if action in valid:
                        break
                    print("  유효하지 않은 칸입니다. 다시 입력하세요.")
                except ValueError:
                    print("  숫자를 입력하세요.")

        state, r_o, r_x, done = env.step(action)

    # 최종 결과
    env.render()
    if r_o == config.REWARD_WIN:
        winner = "O"
    elif r_x == config.REWARD_WIN:
        winner = "X"
    else:
        winner = None

    if winner is None:
        print("  결과: 무승부!")
    elif (winner == "O" and robot_is_o) or (winner == "X" and not robot_is_o):
        print("  결과: 로봇이 이겼습니다!")
    else:
        print("  결과: 당신이 이겼습니다! 로봇을 이기다니 대단합니다!")


# =============================================================================
# 메인: 실행 모드 선택
# =============================================================================

if __name__ == "__main__":
    print("\n실행할 기능을 선택하세요:")
    print("  1. 로봇 성능 평가 (1,000판 자동 대결)")
    print("  2. 사람 vs 로봇 대전")
    choice = input("선택 [1/2]: ").strip()

    if choice == "2":
        play_vs_human()
    else:
        evaluate()
