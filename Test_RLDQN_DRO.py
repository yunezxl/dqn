from Env_Test import Re_Environment
from SolverRLDQN_DRO import RLDQN_DRO
import numpy as np
from statistics import mean, stdev
from math import sqrt
from pandas import DataFrame
import pickle
import os

class load_RLDQN_DRO():
    def __init__(self, CaseParams, TestData, TrainSampleSize, RLDQNHyparams, DisType):
        self.CaseParams = CaseParams
        self.demand = TestData
        self.RLDQNHyparams = RLDQNHyparams
        self.TrainSampleSize = TrainSampleSize
        self.DisType = DisType

        self.p1 = self.CaseParams[0]
        self.p2 = self.CaseParams[1]
        self.s1 = self.CaseParams[2]
        self.l1 = self.CaseParams[3]
        self.s2 = self.CaseParams[4]
        self.l2 = self.CaseParams[5]
        self.c = self.CaseParams[6]

        self.state_size = self.RLDQNHyparams[0]
        self.action_size = self.RLDQNHyparams[1]
        self.gamma = self.RLDQNHyparams[2]
        self.epsilon_decay = self.RLDQNHyparams[3]
        self.epsilon_min = self.RLDQNHyparams[4]
        self.learning_rate = self.RLDQNHyparams[5]
        self.batch_size = self.RLDQNHyparams[6]
        self.update = self.RLDQNHyparams[7]
        self.max_order = self.RLDQNHyparams[1] - 1
        self.UL = self.RLDQNHyparams[8]
        self.DRO_order = self.RLDQNHyparams[9]
        self.DRO_reward = RLDQNHyparams[10]

        self.TIME = len(self.demand[0])
        self.WARMUP = 0

        self.env = Re_Environment(self.state_size, self.demand, self.max_order, self.DRO_order, self.DRO_reward, self.c, self.p1, self.s1, self.l1,
                                  self.p2, self.s2, self.l2, self.TIME, self.WARMUP)

    def Test(self):
        AvgReward_RLDQN_DRO = []
        Feature_Order = []
        agent = RLDQN_DRO(self.CaseParams, [None,None], self.demand, self.RLDQNHyparams, self.DisType)
        agent.epsilon = 0
        RLDQN_name = "RLDQN_DRO_" + str(self.TrainSampleSize) + '_' + self.DisType + '_' + 'C=' + str(self.c) + '.h5'
        RLDQN_folder = 'DQN_Agents'
        Table = 'Table2'
        RLDQN_path = os.path.join(os.getcwd(), Table, RLDQN_folder, RLDQN_name)
        agent.load(RLDQN_path)

        done = False
        state, _ = self.env.reset(WithTeacher=True)
        rewards = []
        Act_Trace = []

        while not done:
            state = np.reshape(state, [1, self.state_size])
            action = agent.act(state)
            Act_Trace.append(action)
            next_state, reward, _, done, _ = self.env.step(action, WithTeacher = True)
            rewards.append(reward)
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state

        for j in range(self.WARMUP):
            rewards.remove(0)
        Act_Trace = Act_Trace[self.WARMUP:]
        avg_per_period = sum(rewards) / (self.TIME - self.WARMUP)
        AvgReward_RLDQN_DRO.append(avg_per_period)
        Feature_Order.append(np.round(np.mean(Act_Trace), 2))


        return AvgReward_RLDQN_DRO, Feature_Order