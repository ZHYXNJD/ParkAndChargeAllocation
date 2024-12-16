import numpy as np
import pandas as pd
from gurobipy import *
import seaborn as sns
import matplotlib.pyplot as plt

# 需求和供给
req_info = pd.read_csv("G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\50-1.csv")
decision_interval = 30
req_info['s_it'] = req_info['arrival_t'] // decision_interval
req_info['e_it'] = req_info['leave_t'] // decision_interval
max_interval = 1440 // decision_interval
req_info = req_info.loc[req_info['e_it'] <= max_interval].reset_index(drop=True)


def get_rmk(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, max(req_info['e_it'])+1))
    for i in range(req_num):
        start = req_info['s_it'].iloc[i]
        end = req_info['e_it'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


req_num = len(req_info)
arr = req_info['s_it']
lea = req_info['e_it']
R_it = get_rmk(req_info).astype(int)  # 需求矩阵
T = R_it.shape[1]  # 时长
pl_num = 50
Q_jt = np.ones((pl_num, T))  # 供应矩阵 1: idle 0: occupied
# register_depart_t = 0
# register_back_t = max_interval

register_depart_t = 360 // decision_interval
register_back_t = 1440 // decision_interval
Q_jt[:, :register_depart_t] = 0  # 6点之前被占用
Q_jt[:, register_back_t:] = 0  # 12点之后被占用


# 假设ps pe pd pb 完全一样 跟时间无关
# ps p-user start early probs
# pe p-user end late probs
# pd o-user departure late probs
# pb o-user back early probs
ps = pe = pd = pb = 0.5


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


# parking probability function of o-user
def p_jt(register_depart_t, register_back_t, t):
    if t == register_depart_t + 1:
        return pd
    elif t == register_back_t - 1:
        return pb
    elif register_depart_t + 2 <= t <= register_back_t - 2:
        return 0
    else:
        return 1


# model
epm = Model("Expectation model")
# decision variable
x_ij = epm.addVars({(i, j) for i in range(req_num) for j in range(pl_num)}, vtype=GRB.BINARY, name='x_ij')
# auxiliary variable
Vt = epm.addVars(T, vtype=GRB.CONTINUOUS, name='Vt')  # overload cost
Ut = epm.addVars(T, vtype=GRB.CONTINUOUS, name='Ut')  # idle cost
epm.update()

# OBJ
alpha = 5
beta = 7.5
gamma = 10
obj = alpha * quicksum(p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num) for j in range(pl_num) for t in
                       range(T)) - beta * quicksum(Vt[t] for t in range(T)) - gamma * quicksum(Ut[t] for t in range(T))

epm.setObjective(obj, GRB.MAXIMIZE)

# constraints
epm.addConstrs(quicksum(x_ij[(i, j)] for j in range(pl_num)) <= 1 for i in range(req_num))
epm.addConstrs(
    quicksum(x_ij[(i, j)] * R_it[i][t] for i in range(req_num)) <= Q_jt[j][t] for j in range(pl_num) for t in range(T))
epm.addConstrs(Vt[t] >= quicksum(quicksum(
    p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num)) + p_jt(register_depart_t, register_back_t, t) for j
                                 in range(pl_num)) - pl_num for t in range(T))
epm.addConstrs(Ut[t] >= pl_num - quicksum(quicksum(
    p_it(arr[i], lea[i], t) * x_ij[(i, j)] for i in range(req_num)) + p_jt(register_depart_t, register_back_t, t) for j
                                          in range(pl_num)) for t in range(T))
epm.addConstrs(Vt[t] >= 0 for t in range(T))
epm.addConstrs(Ut[t] >= 0 for t in range(T))

epm.optimize()

# check the results
total_acc = 0
Q_JT = np.zeros((pl_num,T))
for j in range(pl_num):
    for i in range(req_num):
        if x_ij[(i,j)].X == 1:
            total_acc += 1
            Q_JT[j] += R_it[i]
            print(f"berth:{j} has been assigned to user:{i}")
sns.heatmap(data=Q_JT)
fig, ax = plt.subplots()
ax.bar(x=[t for t in range(T)],height=[Vt[t].X for t in range(T)],label='overload cost',color='orange')
ax.set_ylabel('overload cost')
ax.spines['right'].set_visible(False) # ax右轴隐藏
z_ax = ax.twinx()
z_ax.bar(x=[t for t in range(T)],height=[Ut[t].X for t in range(T)],label='idle cost',color='blue')
z_ax.set_ylabel('idle cost')
ax.legend()
z_ax.legend()
plt.show()
print(f"total acc:{total_acc}")
print('overload cost:')
print([Vt[t].X for t in range(T)])
print('idle cost:')
print([Ut[t].X for t in range(T)])





