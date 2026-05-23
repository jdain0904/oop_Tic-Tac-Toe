# =============================================================================
# train.py — O 로봇과 X 로봇의 자가대전(Self-Play) 학습 루프
#
# [왜 교대 학습을 쓰지 않는가]
#   SWAP_INTERVAL 마다 한쪽을 동결하고 다른 쪽만 학습시키는 교대 학습은
#   "마지막에 배운 쪽이 항상 유리한" last-mover advantage 문제를 만듭니다.
#   훈련이 X 학습 구간으로 끝나면 X가 100% 이기는 편향이 생깁니다.
#
# [채택한 방식] 동시 학습 + 평가 지표 분리
#   두 에이전트가 동시에 배우면서 균형을 유지합니다.
#   훈련 중 통계(epsilon 포함)는 탐험 노이즈가 크므로 개선이 잘 안 보입니다.
#   LOG_INTERVAL 마다 epsilon=0 의 "순수 실력 평가" 를 함께 출력해
#   실제 학습 향상을 눈으로 확인할 수 있게 합니다.
#
# 이 파일을 분리한 이유:
#   - "학습"이라는 단 하나의 책임만 담당합니다.
#   - 학습 완료 후 play.py 에서 결과를 별도로 확인할 수 있도록 역할을 분리합니다.
# =============================================================================

import time
import config
from environment import TicTacToeEnv
from agent import QLearningAgent


def run_episode(env: TicTacToeEnv, agent_o: QLearningAgent, agent_x: QLearningAgent):
    """
    게임 한 판(에피소드)을 진행하고 양쪽 에이전트를 모두 학습시킵니다.
    O는 항상 선공, X는 항상 후공입니다.

    Q-업데이트 원칙 — "상대 응답 후 상태"로 업데이트:
        s'는 내가 다음에 실제로 마주할 상태 = 상대가 응답한 뒤의 state.
    """
    state = env.reset()
    last_o = None
    last_x = None
    result = "draw"

    while not env.done:
        valid = env.get_valid_actions()

        if env.current_player == 1:  # O 차례
            if last_o is not None:
                agent_o.learn(last_o[0], last_o[1], config.REWARD_STEP, state, False, valid)
            action = agent_o.choose_action(state, valid)
            next_state, r_o, r_x, done = env.step(action)
            if r_o == config.REWARD_INVALID:
                agent_o.learn(state, action, r_o, state, False, valid)
                continue
            if done:
                agent_o.learn(state, action, r_o, next_state, True, [])
                if last_x is not None:
                    agent_x.learn(last_x[0], last_x[1], r_x, next_state, True, [])
                result = "O_win" if r_o == config.REWARD_WIN else "draw"
            else:
                last_o = (state, action)

        else:  # X 차례
            if last_x is not None:
                agent_x.learn(last_x[0], last_x[1], config.REWARD_STEP, state, False, valid)
            action = agent_x.choose_action(state, valid)
            next_state, r_o, r_x, done = env.step(action)
            if r_x == config.REWARD_INVALID:
                agent_x.learn(state, action, r_x, state, False, valid)
                continue
            if done:
                agent_x.learn(state, action, r_x, next_state, True, [])
                if last_o is not None:
                    agent_o.learn(last_o[0], last_o[1], r_o, next_state, True, [])
                result = "X_win" if r_x == config.REWARD_WIN else "draw"
            else:
                last_x = (state, action)

        state = next_state

    return result


def evaluate_snapshot(env: TicTacToeEnv, agent_o: QLearningAgent, agent_x: QLearningAgent,
                      n: int = 200) -> float:
    """
    epsilon=0 으로 n 판을 플레이해 무승부 비율을 반환합니다.

    훈련 통계(epsilon 포함)와 별개로 "실제 실력" 을 측정합니다.
    무승부 비율이 높을수록 두 에이전트 모두 최적 전략에 가까워진 것입니다.

    왜 epsilon=0 평가가 필요한가:
        훈련 중에는 무작위 탐험(epsilon)이 섞여 있어서 통계가 노이즈에 묻힙니다.
        epsilon=0 으로 잠시 고정하면 순수하게 배운 전략만 동작해
        실질적인 학습 진척을 확인할 수 있습니다.
    """
    saved_eps_o = agent_o.epsilon
    saved_eps_x = agent_x.epsilon
    agent_o.epsilon = 0.0
    agent_x.epsilon = 0.0

    draws = 0
    for _ in range(n):
        state = env.reset()
        while not env.done:
            valid = env.get_valid_actions()
            if env.current_player == 1:
                action = agent_o.choose_action(state, valid)
            else:
                action = agent_x.choose_action(state, valid)
            state, r_o, r_x, _ = env.step(action)
        if r_o == config.REWARD_DRAW:
            draws += 1

    agent_o.epsilon = saved_eps_o
    agent_x.epsilon = saved_eps_x
    return draws / n * 100


def train():
    """
    O 로봇과 X 로봇을 NUM_EPISODES 판 동안 동시 학습시킵니다.

    로그 형식:
        [에피소드] 훈련통계(epsilon 포함) | 실력평가(epsilon=0)
        훈련통계가 정체처럼 보여도 실력평가 수치가 오르면 학습이 진행 중입니다.
    """
    print("=" * 65)
    print("  4×4 틱택토 강화학습 — O 로봇 vs X 로봇")
    print(f"  총 {config.NUM_EPISODES:,} 판  |  평가 주기 {config.LOG_INTERVAL:,} 판")
    print("=" * 65)
    print(f"  {'에피소드':>8}  {'훈련O승':>6} {'훈련X승':>6} {'훈련무':>6}  "
          f"{'실력평가(ε=0)':>14}  {'ε':>6}")
    print("-" * 65)

    env = TicTacToeEnv()
    agent_o = QLearningAgent("O")
    agent_x = QLearningAgent("X")

    agent_o.load(config.MODEL_PATH_O)
    agent_x.load(config.MODEL_PATH_X)

    counts = {"O_win": 0, "X_win": 0, "draw": 0}
    start_time = time.time()

    for episode in range(1, config.NUM_EPISODES + 1):
        result = run_episode(env, agent_o, agent_x)
        counts[result] += 1
        agent_o.decay_epsilon()
        agent_x.decay_epsilon()

        if episode % config.LOG_INTERVAL == 0:
            total = sum(counts.values())
            o_rate = counts["O_win"] / total * 100
            x_rate = counts["X_win"] / total * 100
            d_rate = counts["draw"]  / total * 100

            # ── 핵심: epsilon=0 으로 실제 실력 평가 ─────────────────────
            eval_draw = evaluate_snapshot(env, agent_o, agent_x, n=400)

            print(
                f"  {episode:>8,}  {o_rate:>5.1f}%  {x_rate:>5.1f}%  {d_rate:>5.1f}%  "
                f"  무승부 {eval_draw:>5.1f}%  "
                f"  ε={agent_o.epsilon:.4f}  {time.time()-start_time:.0f}s"
            )
            counts = {"O_win": 0, "X_win": 0, "draw": 0}

    print("-" * 65)
    print("\n학습 완료! 모델을 저장합니다...")
    agent_o.save(config.MODEL_PATH_O)
    agent_x.save(config.MODEL_PATH_X)
    print(f"총 소요 시간: {time.time()-start_time:.1f}초")
    print("play.py 를 실행하면 학습된 로봇과 대전하거나 평가할 수 있습니다.")


if __name__ == "__main__":
    train()
