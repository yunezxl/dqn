import numpy as np
from scipy.stats import pearsonr
import os
import pickle



def DRO(CaseParams, demand):
    p = [CaseParams[0], CaseParams[1]]
    s = [CaseParams[2], CaseParams[4]]
    l = [CaseParams[3], CaseParams[5]]
    c = CaseParams[6]
    demand_1 = demand[0]
    demand_2 = demand[1]
    rho, _ = pearsonr(demand_1, demand_2)
    mu = [np.mean(demand_1), np.mean(demand_2)]
    sigma = [np.std(demand_1, ddof=1), np.std(demand_2, ddof=1)]
    N = 2
    H2 = 0
    M = 0
    P = 0
    C = 0
    A = 1
    for i in range(0, N):
        H2 = H2 + ((p[i]+l[i]-s[i])**2)*(sigma[i]**2)
        A = A * (p[i]+l[i]-s[i]) * sigma[i]
        M = M + (p[i]+l[i]-s[i])*mu[i]
        P = P + p[i]+l[i]-s[i]
        C = c - s[i]
    H2 = H2 + 2 * A * rho
    B = H2/(M**2)
    K = (P-C)/C
    if B <= K:
        order_FRNV = M / P + ((H2 ** (1 / 2)) / (2 * P)) * (K ** (1 / 2) - (1 / K) ** (1 / 2))
    elif B > K:
        order_FRNV = 0
    DRO_profit = 0
    for cur_d1_index in range(len(demand_1)):
        DRO_profit += p[0] * min(order_FRNV, demand_1[cur_d1_index]) + p[1] * min(order_FRNV, demand_2[cur_d1_index]) + s[0] * max(
            order_FRNV - demand_1[cur_d1_index], 0) + s[1] * max(order_FRNV - demand_2[cur_d1_index], 0) - l[0] * max(
            demand_1[cur_d1_index] - order_FRNV, 0) - l[1] * max(demand_2[cur_d1_index] - order_FRNV, 0) - c * order_FRNV

    return DRO_profit/len(demand_1), order_FRNV

def DRO_Test(CaseParams, Order, TestDemand):
    p1 = CaseParams[0]
    p2 = CaseParams[1]
    s1 = CaseParams[2]
    l1 = CaseParams[3]
    s2 = CaseParams[4]
    l2 = CaseParams[5]
    c = CaseParams[6]

    demand_1 = TestDemand[0]
    demand_2 = TestDemand[1]

    DRO_profit = 0
    for i in range(len(demand_1)):
        DRO_profit += p1 * min(Order, demand_1[i]) +p2 * min(Order, demand_2[i])+ s1 * max(Order - demand_1[i], 0)+ s2 * max(Order - demand_2[i], 0)- l1 * max(demand_1[i]-Order, 0)- l2 * max(demand_2[i]-Order, 0) - c * Order
    return DRO_profit/len(demand_1)

def SaveDRORes(DRO_Results, filename, path):
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(DRO_Results, file)
