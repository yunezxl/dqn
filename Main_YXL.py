from GenerateData import GenerationData
from BootstrapReSample import expand_demand
from SolverDRO import DRO, DRO_Test
from SolverDQN import DQN
from SolverRLDQN_DRO import RLDQN_DRO
from Test_DQN import load_DQN
from Test_RLDQN_DRO import load_RLDQN_DRO

import scipy.stats as stats
import numpy as np
import random
import os
import pickle
import time
import tensorflow as tf

# 开始时间
start_time = time.time()

class TrainAndTest():
    def __init__(self, DisType_Demand, DisParams_Demand, TrainSampleSize, CaseParams, Repeat, Round, Time):
        self.DisType = DisType_Demand
        self.DisParams_Demand = DisParams_Demand
        self.TrainSampleSize = TrainSampleSize
        self.CaseParams = CaseParams
        self.Repeat = Repeat
        self.Round = Round
        self.Time = Time
        self.AvgReward_DRO = []
        self.AvgReward_DQN = []
        self.AvgReward_RLDQN_DRO = []
        self.avgReward_DRO = []
        self.avgReward_DQN = []
        self.avgReward_RLDQN_DRO = []
        self.train_DRO_ord = 0
        self.train_DRO_profit = 0
        self.Order_DRO = []
        self.Order_DQN = []
        self.Order_RLDRO = []
        self.order_DQN = []
        self.order_RLDRO = []
        self.Repeat_DRO = []
        self.Repeat_DQN = []
        self.Repeat_RLDRO = []


    def Train(self):
        TrainSet, _ = GenerationData(self.DisType, self.DisParams_Demand, self.TrainSampleSize)
        DRO_profit, Order_DRO = DRO(self.CaseParams, TrainSet)
        self.train_DRO_ord = Order_DRO
        self.Order_DRO.append(Order_DRO)
        self.train_DRO_profit = DRO_profit

        ExpandDataSet = expand_demand(TrainSet, ExpandSize=10000)

        DQN_Agent = DQN(CaseParams=self.CaseParams, Data=TrainSet, Demand_Train=ExpandDataSet,
                        DQNHyparams=[11, 300, 0.90, 0.98, 0.01, 0.002, 64, 128],
                        DisType=self.DisType)
        DQN_Agent.train()

        RLDQN_DRO_Agent = RLDQN_DRO(CaseParams=self.CaseParams, Data=TrainSet, Demand_Train=ExpandDataSet,
                                    RLDQNHyparams=[13, 300, 0.90, 0.98, 0.01, 0.002, 64, 128, 30, Order_DRO,
                                                   DRO_profit], DisType=self.DisType)
        RLDQN_DRO_Agent.train()


    def CleanStore(self, inner = True):
        if inner:
            self.avgReward_DRO = []
            self.avgReward_DQN = []
            self.avgReward_RLDQN_DRO = []
            self.order_DQN = []
            self.order_RLDRO = []
        else:
            self.AvgReward_DRO = []
            self.AvgReward_DQN = []
            self.AvgReward_RLDQN_DRO = []
            self.Order_DRO = []

    def Test(self):
        _, TestSet = GenerationData(self.DisType, self.DisParams_Demand, self.TrainSampleSize)

        tempReward_DRO = DRO_Test(CaseParams=self.CaseParams, Order=self.train_DRO_ord, TestDemand=TestSet)

        DQN_Agent = load_DQN(CaseParams=self.CaseParams, TestData=TestSet, TrainSampleSize = self.TrainSampleSize,
                             DQNHyparams=[11, 300, 0.90, 0.98, 0.01, 0.002, 64, 128], DisType=self.DisType)     # [11, 300, 0.90, 0.98, 0.01, 0.005, 64, 128]
        DQN_AvgReward, DQN_order = DQN_Agent.Test()
        self.order_DQN.append(DQN_order)


        RLDQN_Agent_DRO = load_RLDQN_DRO(CaseParams=self.CaseParams, TestData=TestSet, TrainSampleSize = self.TrainSampleSize,
                                         RLDQNHyparams=[13, 300, 0.90, 0.98, 0.01, 0.002, 64, 128, 30, self.train_DRO_ord, tempReward_DRO],
                                         DisType=self.DisType)
        result_RLDQN_DRO, RLDQN_DRO_Order = RLDQN_Agent_DRO.Test()
        self.order_RLDRO.append(RLDQN_DRO_Order)

        return [tempReward_DRO, DQN_AvgReward, result_RLDQN_DRO]

    def StoreProfit(self, input, inner = True):
        if inner == True:
            self.avgReward_DRO.append(input[0])
            self.avgReward_DQN.append(input[1])
            self.avgReward_RLDQN_DRO.append(input[2])
        else:
            self.AvgReward_DRO.append(input[0])
            self.AvgReward_DQN.append(input[1])
            self.AvgReward_RLDQN_DRO.append(input[2])

    def StoreOrder(self):
        Mean_Order_DQN = np.mean(self.order_DQN)
        Mean_Order_RLDRO = np.mean(self.order_RLDRO)
        self.Order_DQN.append(Mean_Order_DQN)
        self.Order_RLDRO.append(Mean_Order_RLDRO)

    def AvgPermance(self):
        m1 = np.mean(self.avgReward_DRO)
        m2 = np.mean(self.avgReward_DQN)
        m3 = np.mean(self.avgReward_RLDQN_DRO)
        return [m1, m2, m3]

    def AvgOrder(self):
        o1 = np.mean(self.order_DQN)
        o2 = np.mean(self.order_RLDRO)
        return [o1, o2]

    def Evaluation(self):
        Temp_DQN = (np.array(self.AvgReward_DQN) / np.array(self.AvgReward_DRO) - 1) * 100
        Mean_DQN_Profit = np.round(np.mean(Temp_DQN), 2)
        ci_DQN_Profit = round(1.96 * np.std(Temp_DQN) / np.sqrt(len(Temp_DQN)), 2)

        Temp_RLDRO = (np.array(self.AvgReward_RLDQN_DRO) / np.array(self.AvgReward_DRO) - 1) * 100
        Mean_RLDRO_Profit = np.round(np.mean(Temp_RLDRO), 2)
        ci_RLDRO_Profit = round(1.96 * np.std(Temp_RLDRO) / np.sqrt(len(Temp_RLDRO)), 2)

        Mean_Order_DRO = np.round(np.mean(self.Order_DRO), 2)
        ci_order_DRO = round(1.96 * np.std(self.Order_DRO) / np.sqrt(len(self.Order_DRO)), 2)

        Mean_Order_DQN = np.round(np.mean(self.Order_DQN), 2)
        ci_order_DQN = round(1.96 * np.std(self.Order_DQN) / np.sqrt(len(self.Order_DQN)), 2)

        Mean_Order_RLDRO = np.round(np.mean(self.Order_RLDRO), 2)
        ci_order_RLDRO = round(1.96 * np.std(self.Order_RLDRO) / np.sqrt(len(self.Order_RLDRO)), 2)

        # 重复Repeat次的存储过程
        self.Repeat_DRO.append([round(Mean_Order_DRO), ci_order_DRO, np.round(np.mean(self.AvgReward_DRO), 2)])
        self.Repeat_DQN.append([round(Mean_Order_DQN), ci_order_DQN, np.round(np.mean(self.AvgReward_DQN), 2), Mean_DQN_Profit, ci_DQN_Profit])
        self.Repeat_RLDRO.append([round(Mean_Order_RLDRO), ci_order_RLDRO, np.round(np.mean(self.AvgReward_RLDQN_DRO), 2), Mean_RLDRO_Profit, ci_RLDRO_Profit])


    def Rep_Print(self):
        for i in range(self.Repeat):
            print('***************** 第—— ', i , '——次重复结果 *****************')
            print('DRO订货量 | ', self.Repeat_DRO[i][0], "±", self.Repeat_DRO[i][1], '; DRO平均收益 | ', self.Repeat_DRO[i][2])
            print('DQN订货量 | ', self.Repeat_DQN[i][0], "±", self.Repeat_DQN[i][1], '; DQN平均收益 | ', self.Repeat_DQN[i][2], '; DQN相对收益 | ', self.Repeat_DQN[i][3], "±", self.Repeat_DQN[i][4])
            print('RLDQN_DRO订货量 | ', self.Repeat_RLDRO[i][0], "±", self.Repeat_RLDRO[i][1], '; RLDQN_DRO平均收益 | ', self.Repeat_RLDRO[i][2], '; RLDQN_DRO相对收益 | ', self.Repeat_RLDRO[i][3], "±",
                  self.Repeat_RLDRO[i][4])
            print('*****************')
            print(' ')


    def Todo(self):
        for _ in range(self.Repeat):
            self.CleanStore(inner=False)
            for r in range(self.Round):
                self.Train()
                print("完成第 ", r," 轮训练过程！")
                self.CleanStore(inner=True)
                for t in range(self.Time):
                    TestResult = self.Test()
                    self.StoreProfit(TestResult, inner=True)
                # print("完成 ", self.Time," 轮测试过程！")
                AvgTProfit = self.AvgPermance()
                AvgTOrder = self.AvgOrder()
                self.StoreProfit(AvgTProfit, inner=False)
                self.StoreOrder()
            self.Evaluation()
        self.Rep_Print()



# ——实例计算——————————————————————————————————————————————————————————————————————————————————————————————————————————— #
# 生成测试类的实例
# A_Case = TrainAndTest("Normal",
#                       [[100, 140], [[25 ** 2, 0.5*25*35], [0.5*25*35, 35 ** 2]]],
#                       10, [40, 60, 4, 4, 6, 6, 50],
#                       Repeat = 1, Round = 10, Time = 1)  # Round外层循环； Time内层循环
# A_Case = TrainAndTest("CompoundNormal",
#                       [[100, 140], [[25 ** 2, 0.5*25*35], [0.5*25*35, 35 ** 2]]],
#                       10, [40, 60, 4, 4, 6, 6, 50],
#                       Repeat = 2, Round = 10, Time = 1)
# A_Case = TrainAndTest("Mix",
#                       [[[100, 140], [[25 ** 2, 0.5*25*35], [0.5*25*35, 35 ** 2]]],
#                        [100, 140],
#                        [[0, 0], [200, 280]]],
#                       10, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 3, Round = 10, Time = 1)
# A_Case = TrainAndTest("Cyclic",
#                       [[[100,140], [[25 ** 2, 0.5*25*35], [0.5*25*35, 35 ** 2]]], [100,140]],
#                       10, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 1, Round = 100, Time = 1)
A_Case = TrainAndTest("Exponential",
                      [100,140],
                      40, [40, 60, 4, 4, 6, 6, 20],
                      Repeat = 1, Round = 100, Time = 1)
# A_Case = TrainAndTest("Gamma",
#                       [[16, 16], [6, 8]],
#                       10, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 1, Round = 5, Time = 1)
# A_Case = TrainAndTest("LogNormal",
#                       [[5, 5], [0.15, 0.3]],
#                       10, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 1, Round = 10, Time = 1)
# A_Case = TrainAndTest("RealData", None, 20, [40, 60, 4, 4, 6, 6, 20],
#                       Repeat = 1, Round = 10, Time = 1)             # 13695

A_Case.Todo()




# 结束时间
end_time = time.time()
# 计算运行时间
run_time = end_time - start_time
# 打印开始和结束时间
print("开始时间:", time.ctime(start_time))
print("结束时间:", time.ctime(end_time))
# 打印运行时间
print("运行时间:", run_time, "秒")