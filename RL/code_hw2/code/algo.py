import numpy as np
import random
from abc import abstractmethod
from collections import defaultdict
class QAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass
# class MyQAgent(QAgent):
#     def __init__(self,alpha=0.5, gamma=0.8):
#         super().__init__()
#         self.learningRate = alpha
#         self.discountFactor = gamma
#         self.qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
#
#     def select_action(self, ob):
#         stateQ = self.qTable[str(ob)]
#         max = []
#         maxValue = stateQ[0]
#         max.append(0)
#         for i in range(1, 4):
#             if stateQ[i] > maxValue:
#                 max.clear()
#                 maxValue = stateQ[i]
#                 max.append(i)
#             elif stateQ[i] == maxValue:
#                 max.append(i)
#         return random.choice(max)
#
#     def learn(self, ob, action, reward, nextOb):
#         oldQ = self.qTable[str(ob)][action]
#         newQ = reward + self.discountFactor * max(self.qTable[str(nextOb)])
#         self.qTable[str(ob)][action] += self.learningRate * (newQ - oldQ)


class MyQAgent(QAgent):
    def __init__(self, action_shape=4, observation_shape=2,alpha=0.5, gamma=0.9):
        super().__init__()
        # 动作空间维度
        self.action_shape = action_shape
        self.observation_shape = observation_shape
        # 学习率
        self.alpha = alpha
        # 折扣因子
        self.gamma = gamma
        # 使用 defaultdict 初始化 Q 表，键为状态，值为动作 Q 值列表
        self.q_table = defaultdict(lambda: [0.1 for _ in range(action_shape)])

    def select_action(self, state):
        """
        使用贪婪策略选择动作，如果存在多个最大值则随机选择其中一个。
        """
        state_str = str(state)
        state_q = self.q_table[state_str]
        max_actions = []
        max_value = state_q[0]
        max_actions.append(0)
        # 找到最大 Q 值的所有动作
        for i in range(1, self.action_shape):
            if state_q[i] > max_value:
                max_actions.clear()
                max_value = state_q[i]
                max_actions.append(i)
            elif state_q[i] == max_value:
                max_actions.append(i)
        # 从最大 Q 值的动作中随机选择一个
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state):
        """
        使用 Q-learning 算法更新 Q 值。
        """
        state_str = str(state)
        next_state_str = str(next_state)
        # 当前状态的 Q 值
        old_q = self.q_table[state_str][action]
        # 计算最优的未来 Q 值
        max_future_q = max(self.q_table[next_state_str]) if reward != 100 else 0  # 终止状态未来 Q 值为 0
        # 更新 Q 值
        new_q = reward + self.gamma * max_future_q
        self.q_table[state_str][action] += self.alpha * (new_q - old_q)
#
