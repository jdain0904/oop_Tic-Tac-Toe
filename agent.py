# =============================================================================
# agent.py — Q-러닝 에이전트 (O 로봇 / X 로봇 공통 클래스)
#
# Q-러닝(Q-Learning)이란?
#   각 상황(state)에서 각 행동(action)이 얼마나 좋은지를 숫자(Q-값)로 기억하고,
#   경험을 쌓으면서 그 숫자를 점점 정확하게 업데이트해 나가는 학습 방법입니다.
#
#   Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
#   │            │             │    │      └── 다음 상태에서 최선의 미래 가치
#   │            │             │    └── 할인율(γ): 미래 가치를 현재 기준으로 환산
#   │            │             └── 받은 보상(r)
#   │            └── 기존 Q-값
#   └── 업데이트된 Q-값
#
# 이 파일을 에이전트 전용으로 분리한 이유:
#   - O 로봇과 X 로봇은 같은 학습 알고리즘을 쓰지만 서로 다른 Q-테이블을 가집니다.
#     같은 클래스에서 인스턴스만 두 개 만들면 코드 중복 없이 구현됩니다.
#   - 알고리즘 변경(예: Q-러닝 → SARSA) 시 이 파일만 수정하면 됩니다.
#   - 환경(environment.py)과 에이전트(agent.py) 역할을 분리해 각자 독립 테스트가 가능합니다.
# =============================================================================

import gzip
import pickle
import random
import os
import config


class QLearningAgent:
    """
    Q-러닝 기반 틱택토 에이전트.

    Q-테이블(q_table):
        {상태(state): {행동(action): Q-값}} 형태의 딕셔너리.
        처음 본 상태는 자동으로 0.0 으로 초기화됩니다.
        3×3 틱택토의 가능한 상태는 약 5,478가지이므로 메모리 걱정 없이 모두 저장합니다.
    """

    def __init__(self, name: str):
        """
        name: 에이전트 이름 (예: "O", "X") — 로그 출력 시 구별용
        """
        self.name = name

        # Q-테이블: 상태 → {행동 → Q-값} 의 중첩 딕셔너리
        # defaultdict 대신 일반 dict 를 사용해 저장·불러오기를 단순하게 유지합니다.
        self.q_table: dict[tuple, dict[int, float]] = {}

        # ε(epsilon): 현재 탐험 비율. 학습이 진행될수록 점점 줄어듭니다.
        self.epsilon = config.EPSILON_START

    # ─────────────────────────────────────────────────────────────────────────
    # choose_action: 상태와 유효한 행동 목록을 받아 행동 하나를 선택합니다.
    #
    # ε-greedy 전략:
    #   ε 확률로 무작위 행동(탐험) → 아직 모르는 수를 시도해봄
    #   1-ε 확률로 최선의 행동(활용) → 지금까지 배운 가장 좋은 수를 선택
    # ─────────────────────────────────────────────────────────────────────────
    def choose_action(self, state: tuple, valid_actions: list[int]) -> int:
        """
        state       : 현재 보드 상태 (9개 값의 tuple)
        valid_actions: 둘 수 있는 칸 번호 목록
        반환값      : 선택된 칸 번호 (0~8)
        """
        # 탐험: ε 확률로 유효한 칸 중 무작위 선택
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 활용: Q-값이 가장 높은 행동을 선택
        # 아직 경험하지 못한 행동은 Q-값 0.0 으로 간주합니다.
        q_values = self._get_q_values(state)
        best_action = max(valid_actions, key=lambda a: q_values.get(a, 0.0))
        return best_action

    # ─────────────────────────────────────────────────────────────────────────
    # learn: 경험(s, a, r, s', done)을 바탕으로 Q-값을 업데이트합니다.
    #
    # 벨만 방정식(Bellman Equation)을 한 스텝씩 적용합니다:
    #   새 Q-값 = 기존 Q-값 + α × (TD-목표 − 기존 Q-값)
    #   TD-목표 = r + γ × max Q(s', a')   (게임 종료 시에는 r 만)
    # ─────────────────────────────────────────────────────────────────────────
    def learn(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
        next_valid_actions: list[int],
    ):
        """
        state            : 행동 전 보드 상태
        action           : 선택한 칸 번호
        reward           : 이 행동으로 받은 보상
        next_state       : 행동 후 보드 상태
        done             : 게임 종료 여부
        next_valid_actions: 다음 상태에서 둘 수 있는 칸 목록 (종료 시 빈 리스트)
        """
        q_values = self._get_q_values(state)
        current_q = q_values.get(action, 0.0)

        if done or not next_valid_actions:
            # 게임이 끝났으면 미래 가치 없이 보상만 목표로 삼습니다.
            td_target = reward
        else:
            next_q_values = self._get_q_values(next_state)
            # 다음 상태에서 유효한 행동 중 최대 Q-값을 미래 가치로 사용합니다.
            max_next_q = max(next_q_values.get(a, 0.0) for a in next_valid_actions)
            td_target = reward + config.DISCOUNT_FACTOR * max_next_q

        # Q-값 업데이트: 기존값과 목표값의 차이를 학습률만큼 반영합니다.
        new_q = current_q + config.LEARNING_RATE * (td_target - current_q)
        self.q_table.setdefault(state, {})[action] = new_q

    # ─────────────────────────────────────────────────────────────────────────
    # decay_epsilon: 탐험 비율을 한 스텝 감소시킵니다.
    # 매 에피소드(게임 1판) 종료 후 호출합니다.
    # ─────────────────────────────────────────────────────────────────────────
    def decay_epsilon(self):
        """ε 을 EPSILON_DECAY 비율로 줄이고, EPSILON_END 이하로 내려가지 않게 합니다."""
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    # ─────────────────────────────────────────────────────────────────────────
    # save / load: 학습된 Q-테이블을 파일로 저장하거나 불러옵니다.
    # pickle 은 파이썬 객체를 그대로 파일에 쓰는 직렬화 방식입니다.
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        """Q-테이블과 epsilon 을 gzip 압축 pkl 파일로 저장합니다."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=9) as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[{self.name}] 모델 저장 완료 → {path}")

    def load(self, path: str):
        """저장된 gzip 압축 pkl 파일에서 Q-테이블과 epsilon 을 불러옵니다."""
        if not os.path.exists(path):
            print(f"[{self.name}] 저장 파일 없음 — 처음부터 학습합니다.")
            return
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"[{self.name}] 모델 불러오기 완료 ← {path}  (ε={self.epsilon:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _get_q_values(self, state: tuple) -> dict[int, float]:
        """해당 상태의 Q-값 딕셔너리를 반환합니다. 없으면 빈 딕셔너리를 등록합니다."""
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state]
