from Env_Test import Re_Environment
from SolverDQN import DQN
import numpy as np
from statistics import mean, stdev
import scipy.stats as stats
from math import sqrt
from pandas import DataFrame
import pickle
import os

class load_DQN():
    def __init__(self, CaseParams, TestData, TrainSampleSize, DQNHyparams, DisType):
        self.CaseParams = CaseParams
        self.demand = TestData
        self.DQNHyparams = DQNHyparams
        self.TrainSampleSize = TrainSampleSize
        self.DisType = DisType

        self.p1 = self.CaseParams[0]
        self.p2 = self.CaseParams[1]
        self.s1 = self.CaseParams[2]
        self.l1 = self.CaseParams[3]
        self.s2 = self.CaseParams[4]
        self.l2 = self.CaseParams[5]
        self.c = self.CaseParams[6]

        self.state_size = self.DQNHyparams[0]
        self.action_size = self.DQNHyparams[1]
        self.gamma = self.DQNHyparams[2]
        self.epsilon_decay = self.DQNHyparams[3]
        self.epsilon_min = self.DQNHyparams[4]
        self.learning_rate = self.DQNHyparams[5]
        self.batch_size = self.DQNHyparams[6]
        self.update = self.DQNHyparams[7]
        self.max_order = self.DQNHyparams[1] - 1

        self.TIME = len(self.demand[0])
        self.WARMUP = 0

        self.env = Re_Environment(self.state_size, self.demand, self.max_order, None, None, self.c, self.p1, self.s1, self.l1,
                                  self.p2, self.s2, self.l2, self.TIME, self.WARMUP)

    def Test(self):
        AvgReward_DQN = []
        Feature_Order = []
        agent = DQN(self.CaseParams, [None,None], self.demand, self.DQNHyparams, self.DisType)
        agent.epsilon = 0
        DQN_name = "DQN_Agent_" + str(self.TrainSampleSize) + '_' + self.DisType + '_' + 'C=' + str(self.c) + '.h5'
        DQN_folder = 'DQN_Agents'
        Table = 'Table2'
        DQN_path = os.path.join(os.getcwd(), Table, DQN_folder, DQN_name)
        agent.load(DQN_path)

        done = False
        state, _ = self.env.reset(WithTeacher=False)
        rewards = []
        Act_Trace = []

        while not done:
            state = np.reshape(state, [1, self.state_size])
            action = agent.act(state)
            Act_Trace.append(action)
            next_state, reward, done, _ = self.env.step(action, WithTeacher = False)
            rewards.append(reward)
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state

        for j in range(self.WARMUP):
            rewards.remove(0)
        Act_Trace = Act_Trace[self.WARMUP:]
        avg_per_period = sum(rewards) / (self.TIME - self.WARMUP)
        AvgReward_DQN.append(avg_per_period)
        Feature_Order.append(np.round(np.mean(Act_Trace), 2))

        return AvgReward_DQN, Feature_Order