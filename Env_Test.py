import numpy as np
import copy
import random
import math

class Re_Environment(object):

    def __init__(self, state_size, demand_samples,
                 max_order, teach_order, teach_reward,
                 cost_order, revenue_sell1, revenue_salvage1, cost_stockout1,
                 revenue_sell2, revenue_salvage2, cost_stockout2, time, warmup):

        self.state_size = state_size
        self.demand1 = demand_samples[0]
        self.demand2 = demand_samples[1]

        self.teach_order = teach_order
        self.teach_reward = teach_reward
        self.max_order = max_order
        self.cost_order = cost_order
        self.revenue_sell1 = revenue_sell1
        self.revenue_salvage1 = revenue_salvage1
        self.cost_stockout1 = cost_stockout1
        self.revenue_sell2 = revenue_sell2
        self.revenue_salvage2 = revenue_salvage2
        self.cost_stockout2 = cost_stockout2
        self.time = time
        self.warmup = warmup

        self.i_round = 0
        self.action = 0
        self.current_time = 0
        self.reward = 0
        self.reward_after = 0

        self.state = []
        for i in range(self.state_size):
            self.state.append(0)

        self.render_state = self.state.copy()

        self.action_space = []
        for i in range(self.max_order + 1):
            self.action_space.append(i)

        np.random.seed(10)

        # print('Environment created...')

    def step(self, action, WithTeacher):

        self.action = action

        demand1 = self.demand1[self.current_time]
        demand2 = self.demand2[self.current_time]

        CumTrainReward = 0
        for ith_tr in range(len(self.demand1)):
            agent_order_cost = self.action * self.cost_order
            agent_sell_revenue = min(self.action, self.demand1[ith_tr]) * self.revenue_sell1 + min(self.action, self.demand2[ith_tr]) * self.revenue_sell2
            agent_salvage_revenue = max(self.action - self.demand1[ith_tr], 0) * self.revenue_salvage1 + max(self.action - self.demand2[ith_tr],
                                                                                     0) * self.revenue_salvage2
            agent_stockout_cost = max(self.demand1[ith_tr] - self.action, 0) * self.cost_stockout1 + max(self.demand2[ith_tr] - self.action,
                                                                                 0) * self.cost_stockout2
            CumTrainReward += -agent_order_cost + agent_sell_revenue + agent_salvage_revenue - agent_stockout_cost


        if self.current_time >= self.warmup:
            self.reward = CumTrainReward / len(self.demand1)
            if WithTeacher:
                if self.teach_reward > self.reward:
                    self.reward_after = self.reward - 1 * (
                                self.reward - self.teach_reward) **2
                else:
                    self.reward_after = self.reward + 1 * (
                                self.reward - self.teach_reward)**2
        else:
            self.reward = 0
        self.state[0] = demand1
        self.state[1] = demand2
        self.state[2] = action
        if WithTeacher:
            self.state[3] = self.reward_after
        else:
            self.state[3] = self.reward

        self.current_time += 1

        if WithTeacher:
            return self.state, self.reward, self.reward_after, self.isFinished(self.current_time), None
        else:
            return self.state, self.reward, self.isFinished(self.current_time), None

    def isFinished(self, current_time):
        return current_time == self.time

    def reset(self, WithTeacher):
        self.current_time = 0

        self.state = [0 for _ in range(self.state_size)]
        if WithTeacher:
            self.state[4] = self.teach_order
            self.state[5] = self.teach_reward
            self.state[6] = self.revenue_sell1
            self.state[7] = self.revenue_sell2
            self.state[8] = self.revenue_salvage1
            self.state[9] = self.revenue_salvage2
            self.state[10] = self.cost_stockout1
            self.state[11] = self.cost_stockout2
            self.state[12] = self.cost_order
        else:
            self.state[4] = self.revenue_sell1
            self.state[5] = self.revenue_sell2
            self.state[6] = self.revenue_salvage1
            self.state[7] = self.revenue_salvage2
            self.state[8] = self.cost_stockout1
            self.state[9] = self.cost_stockout2
            self.state[10] = self.cost_order
        #print('Reset environment...')
        return self.state, self.current_time

    def random_action(self):
        return random.sample(self.action_space, 1)[0]