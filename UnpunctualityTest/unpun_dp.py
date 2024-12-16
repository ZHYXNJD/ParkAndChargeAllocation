
from gurobipy import *
import seaborn as sns
import matplotlib.pyplot as plt
from strategy.utils import *
from entity import OD

np.random.seed(100)

# 需求和供给
req_info = pd.read_csv("G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\250-1.csv")
decision_interval = 15
req_info['request_interval'] = req_info['request_t'] // decision_interval
req_info['s_it'] = req_info['arrival_t'] // decision_interval
req_info['e_it'] = req_info['leave_t'] // decision_interval
request_interval = req_info['request_interval'].unique()
req_info[['revenue', 'std_revenue']] = pd.Series(
    get_revenue(req_info['activity_t'].values, req_info['new_label'].values))

# 15分钟的费用
park_fee = 5
fast_charge_fee = 10
slow_charge_fee = 5
fee_li = [park_fee, fast_charge_fee, slow_charge_fee]

# basic info
# index
N = 100
K = max(req_info['e_it'])
request_interval.sort()
I = request_interval
O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
cost_matrix_ = OD.OdCost().cost_matrix

# 泊位索引
T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
_, OPS_INDEX = P_ZN(Z, N, pl)
_, CPS_INDEX = C_ZN(Z, N, pl)
_, FAST_INDEX = Fast_ZN(Z, N, pl)
_, SLOW_INDEX = Slow_ZN(Z, N, pl)


def get_rmk(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, max(req_info['e_it']) + 1))
    for i in range(req_num):
        start = req_info['s_it'].iloc[i]
        end = req_info['e_it'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


req_num = len(req_info)
arr = req_info['s_it']
lea = req_info['e_it']

# S_NK = np.ones((len(I) + 1, N, K + 1)).astype(int)
S_NK = np.ones((N, K + 1)).astype(int)

r_mk = get_rmk(req_info).astype(int)

ps = pe = pd = pb = 0.2


# parking probability function of p-user
def p_it(register_arrival_t, register_leave_t, t):
    if t == register_arrival_t - 1:
        return ps
    elif t == register_leave_t + 1:
        return pe
    elif register_arrival_t <= t <= register_leave_t:
        return 1
    else:
        return 0


# 预先计算pit 构建pit矩阵
P_IT = {(m, t): p_it(arr[m], lea[m], t) for t in range(K) for m in range(req_num)}
# XNM记录分配结果
X_NM = {(n, m): 0 for n in range(N) for m in range(req_num)}

# Update PJT
# 初始化pjt全部为0  表示某泊位初始时刻均可停车
# 此后每有用户停放 则更新每个时刻的概率
P_JT = np.zeros((N, K + 2))


def update_pjt(j, register_arrival_t, register_leave_t):
    """
    j: user j is assigned to berth n
    """
    P_JT[j][register_arrival_t - 1] += ps
    P_JT[j][register_leave_t + 1] += pe
    P_JT[j][register_arrival_t:register_leave_t + 1] += 1


rule = 2
total_acc = 0
park_rev = 0
char_rev = 0
park_re = 0
char_re = 0
total_re = 0
travel_t = 0


for ith, i in enumerate(I):

    epm = Model("Expectation model")

    temp_req_info = req_info[req_info['request_interval'] == i]
    total_index = temp_req_info.index.tolist()
    park_index = temp_req_info[temp_req_info['label'] == 0].index.tolist()
    charge_index = temp_req_info[temp_req_info['label'] == 1].index.tolist()
    fast_charge_index = temp_req_info[temp_req_info['new_label'] == 1].index.tolist()
    slow_charge_index = temp_req_info[temp_req_info['new_label'] == 2].index.tolist()

    # X_NM = np.zeros((N, len(temp_req_info))).astype(int)  # 该请求的分配情况
    # R_MK = r_mk[total_index].reshape((len(temp_req_info), K))  # 该请求

    x_nm = epm.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
    y_mz = epm.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

    Vt = epm.addVars(K, vtype=GRB.CONTINUOUS, name='Vt')  # overload cost
    Ut = epm.addVars(K, vtype=GRB.CONTINUOUS, name='Ut')  # idle cost

    epm.update()
    """
    obj1: 收益 - 运营费用（空闲费用 + 过载费用（不准时情况考虑）） 平台的角度
    obj2: - 平均出行成本                                   用户的角度
    obj3: - 拒绝惩罚                                      系统的角度
    """
    # obj1: 收益： 预约费用 + 停车费用 + 充电费用 - 空闲费用 - 过载费用
    overload_coeff = 5
    idle_coeff = 5

    obj1 = quicksum(
        P_IT[(m, t)] * x_nm[(n, m)] * fee_li[req_info['new_label'].loc[m]] for m in total_index for n in
        range(N) for t in range(K)) - overload_coeff * quicksum(Vt[t] for t in range(K)) - idle_coeff * quicksum(
        Ut[t] for t in range(K))

    park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
    char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))

    # obj2: 平均出行成本
    obj2 = quicksum(
        cost_matrix_[z][temp_req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][
            temp_req_info['D'].loc[m] + 2] * y_mz[
            (m, z)] for m in total_index for z in range(Z))

    # obj3: 平台拒绝数量
    # 拒绝的停车数量
    refuse_coeff = 8
    refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
    refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
    obj3 = refuse_coeff * (refuse_park + refuse_char)

    # 总目标
    obj = obj1 - obj2 - obj3
    epm.setObjective(obj, GRB.MAXIMIZE)

    # 约束条件
    epm.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
    epm.addConstrs(quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= S_NK[n][k] for n in range(N) for k in range(K))
    epm.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in total_index)

    epm.addConstrs(Vt[t] >= quicksum(quicksum(
        P_IT[(m, t)] * x_nm[(n, m)] for m in total_index) + P_JT[n][t] for n in range(N)) - N for t in
                   range(K))

    # epm.addConstrs(
    #     Nt[(n, t)] >= quicksum(P_IT[(m, t)] * x_nm[(n, m)] for m in total_index) + P_JT[n][t] - 1 for n in range(N) for
    #     t in range(K))
    # epm.addConstrs(Nt[(n, t)] >= 0 for n in range(N) for t in range(K))
    # epm.addConstrs(Vt[t] >= quicksum(Nt[(n, t)] for n in range(N)) for t in range(K))

    epm.addConstrs(Ut[t] >= N - quicksum(quicksum(
        P_IT[(m, t)] * x_nm[(n, m)] for m in total_index) + P_JT[n][t] for n in range(N)) for t in range(K))

    epm.addConstrs(Vt[t] >= 0 for t in range(K))
    epm.addConstrs(Ut[t] >= 0 for t in range(K))

    if rule == 1:
        # rule1
        # 停车请求只能分配到OPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in park_index)
        epm.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
    elif rule == 2:
        # rule2:
        # 停车请求可以分配到OPS和CPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in park_index)
        pass
    # 快充电请求只能分配到快充CPS
    # m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) <= 1 for m in fast_charge_index)
    epm.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
    epm.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
    # 慢充电请求只能分配到慢充CPS
    # m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) <= 1 for m in slow_charge_index)
    epm.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
    epm.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

    epm.optimize()

    for n in range(N):
        for m in total_index:
            if x_nm[(n, m)].X == 1:
                total_acc += 1
                S_NK[n] -= r_mk[m]
                X_NM[(n, m)] = 1
                update_pjt(n, register_arrival_t=req_info['s_it'].loc[m], register_leave_t=req_info['e_it'].loc[m])
                print(f"berth:{n} has been assigned to user:{m}")

    # S_NK[ith + 1] += S_NK[ith]

    park_rev += park_revenue.getValue()
    char_rev += char_revenue.getValue()
    park_re += refuse_park.getValue()
    char_re += refuse_char.getValue()
    total_re += obj3.getValue() / refuse_coeff
    travel_t += obj2.getValue()

    if ith == len(I) - 1:
        print(f"total acc:{total_acc}")
        print('overload cost:')
        print([Vt[t].X for t in range(K)])
        # oc = [sum(max(P_JT[n][t] - 1, 0) for n in range(N)) for t in range(K)]
        # print(oc)
        print('idle cost:')
        # ic = [N - sum(P_JT[n][t] for n in range(N)) for t in range(K)]
        # print(ic)
        print([Ut[t].X for t in range(K)])
        print(f"park revenue:{park_rev}")
        print(f"charge revenue:{char_rev}")
        print(f"park refuse:{park_re}")
        print(f"charge refuse:{char_re}")
        print(f"travel time:{travel_t / (len(req_info) - total_re)}")
        fig, ax = plt.subplots(1,3)
        ax[0].bar(x=[t for t in range(K)], height=[Vt[t].X for t in range(K)], label='overload cost', color='orange')
        ax[0].set_ylabel('overload cost')
        ax[0].spines['right'].set_visible(False)  # ax右轴隐藏
        z_ax = ax[0].twinx()
        z_ax.bar(x=[t+5 for t in range(K)], height=[Ut[t].X for t in range(K)], label='idle cost', color='blue')
        z_ax.set_ylabel('idle cost')
        ax[0].legend()
        z_ax.legend()
        sns.heatmap(data=S_NK,ax=ax[1])
        ax[1].set_title('SNK')
        sns.heatmap(data=P_JT,ax=ax[2])
        ax[2].set_title('PJT')
        plt.show()

# R_it = get_rmk(req_info).astype(int)  # 需求矩阵
# T = R_it.shape[1]  # 时长
# pl_num = 15
# Q_jt = np.ones((pl_num, T))  # 供应矩阵 1: idle 0: occupied
# # register_depart_t = 0
# # register_back_t = max_interval
#
# register_depart_t = 360 // decision_interval
# register_back_t = 1440 // decision_interval
# Q_jt[:, :register_depart_t] = 0  # 6点之前被占用
# Q_jt[:, register_back_t:] = 0  # 12点之后被占用
#
#
# # 假设ps pe pd pb 完全一样 跟时间无关
# # ps p-user start early probs
# # pe p-user end late probs
# # pd o-user departure late probs
# # pb o-user back early probs
# ps = pe = pd = pb = 0.5
#
#
# # parking probability function of p-user
# def p_it(register_arrival_t, register_leave_t, t):
#     if t == register_arrival_t - 1:
#         return ps
#     elif t == register_leave_t + 1:
#         return pe
#     elif register_arrival_t <= t <= register_leave_t:
#         return 1
#     else:
#         return 0
#
#
# # parking probability function of o-user
# def p_jt(register_depart_t, register_back_t, t):
#     if t == register_depart_t + 1:
#         return pd
#     elif t == register_back_t - 1:
#         return pb
#     elif register_depart_t + 2 <= t <= register_back_t - 2:
#         return 0
#     else:
#         return 1
#
#
# # model
# epm = Model("Expectation model")
# # decision variable
# x_ij = epm.addVars({(i, j) for i in range(req_num) for j in range(pl_num)}, vtype=GRB.BINARY, name='x_ij')
# # auxiliary variable
# Vt = epm.addVars(T, vtype=GRB.CONTINUOUS, name='Vt')  # overload cost
# Ut = epm.addVars(T, vtype=GRB.CONTINUOUS, name='Ut')  # idle cost
# epm.update()
#
# # OBJ
# alpha = 5
# beta = 7.5
# gamma = 10
# obj = alpha * quicksum(p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num) for j in range(pl_num) for t in
#                        range(T)) - beta * quicksum(Vt[t] for t in range(T)) - gamma * quicksum(Ut[t] for t in range(T))
#
# epm.setObjective(obj, GRB.MAXIMIZE)
#
# # constraints
# epm.addConstrs(quicksum(x_ij[(i, j)] for j in range(pl_num)) <= 1 for i in range(req_num))
# epm.addConstrs(
#     quicksum(x_ij[(i, j)] * R_it[i][t] for i in range(req_num)) <= Q_jt[j][t] for j in range(pl_num) for t in range(T))
# epm.addConstrs(Vt[t] >= quicksum(quicksum(
#     p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num)) + p_jt(register_depart_t, register_back_t, t) for j
#                                  in range(pl_num)) - pl_num for t in range(T))
# epm.addConstrs(Ut[t] >= pl_num - quicksum(quicksum(
#     p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num)) + p_jt(register_depart_t, register_back_t, t) for j
#                                           in range(pl_num)) for t in range(T))
# epm.addConstrs(Vt[t] >= 0 for t in range(T))
# epm.addConstrs(Ut[t] >= 0 for t in range(T))
#
# epm.optimize()
#
# # check the results
# total_acc = 0
# Q_JT = np.zeros((pl_num,T))
# for j in range(pl_num):
#     for i in range(req_num):
#         if x_ij[(i,j)].X == 1:
#             total_acc += 1
#             Q_JT[j] += R_it[i]
#             print(f"berth:{j} has been assigned to user:{i}")
# sns.heatmap(data=Q_JT)
# fig, ax = plt.subplots()
# ax.bar(x=[t for t in range(T)],height=[Vt[t].X for t in range(T)],label='overload cost',color='orange')
# ax.set_ylabel('overload cost')
# ax.spines['right'].set_visible(False) # ax右轴隐藏
# z_ax = ax.twinx()
# z_ax.bar(x=[t for t in range(T)],height=[Ut[t].X for t in range(T)],label='idle cost',color='blue')
# z_ax.set_ylabel('idle cost')
# ax.legend()
# z_ax.legend()
# plt.show()
# print(f"total acc:{total_acc}")
# print('overload cost:')
# print([Vt[t].X for t in range(T)])
# print('idle cost:')
# print([Ut[t].X for t in range(T)])
