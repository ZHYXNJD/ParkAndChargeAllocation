
from gurobipy import *
import seaborn as sns
import matplotlib.pyplot as plt

from entity import OD
from strategy.utils import *

# 需求
park_num = 300
charge_ratio = 1
req_info = get_request(park_num=park_num, charge_ratio=charge_ratio)
# index
total_index, park_index, charge_index, fast_charge_index, slow_charge_index = get_index(req_info)

I = len(req_info)
N = 100

# basic info
O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
cost_matrix_ = OD.OdCost().cost_matrix
cost_matrix = OD.OdCost().get_std_cost()

# 泊位索引
T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
_, OPS_INDEX = P_ZN(Z, N, pl)
_, CPS_INDEX = C_ZN(Z, N, pl)
_, FAST_INDEX = Fast_ZN(Z, N, pl)
_, SLOW_INDEX = Slow_ZN(Z, N, pl)

REVENUE_total = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0目标函数 1总收益 2停车收益 3充电收益 4拒绝总数 5停车拒绝数 6充电拒绝数 7出行时间 8巡游时间


# calculate user's actual arrival and departure time
# this can be learned from history records
# for simple we assume the distribution if independent of arrival and departure time
# for short time activity,the arrival and departure unpunctuality sigma follows the uniform distribution of 0-6 min
# for long time activity,the arrival and departure unpunctuality sigma follows the uniform distribution of 5-25 min
def actual_t(activity_t, s_i, e_i):
    """
    s_i: submitted start time
    e_i: submitted end time
    """
    if activity_t <= 120:
        sigma = np.random.uniform(low=0, high=6)
    else:
        sigma = np.random.uniform(low=5, high=25)

    actual_s_i = np.random.normal(loc=s_i, scale=sigma)
    actual_e_i = np.random.normal(loc=e_i, scale=sigma)

    return actual_s_i, actual_e_i


req_info[['actual_s', 'actual_e']] = req_info.apply(
    lambda x: pd.Series(actual_t(x['activity_t'], x['arrival_t'], x['leave_t'])),
    axis=1
)

req_info[['actual_s', 'actual_e']] = req_info[['actual_s', 'actual_e']].astype(int)
req_info['actual_activity'] = req_info['actual_e'] - req_info['actual_s']
req_info[['actual_rev', 'std_actual_rev']] = pd.Series(
    get_revenue(req_info['actual_activity'].values, req_info['new_label'].values))
req_info[['rev', 'std_rev']] = pd.Series(get_revenue(req_info['activity_t'].values, req_info['new_label'].values))

arr = req_info['actual_s']
lea = req_info['actual_e']
activity = req_info['actual_activity']

plp = Model("linearized model")

# decision variable
# x_ij = plp.addVars({(i, j) for i in range(I) for j in range(J)}, vtype=GRB.BINARY, name='x_ij')
x_nm = plp.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name='x_nm')
y_mz = plp.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

plp.update()


# auxiliary variable
def z_ik(s_i, e_k, buffer=10):
    if s_i - e_k >= buffer:
        return 1
    else:
        return 0


Z_IK = {(i, k): z_ik(arr[i], lea[k]) for i in range(I) for k in range(I) if i != k}

# 平台收益： 预约费用 + 停车费用 + 充电费用
park_revenue = quicksum(x_nm[(n, m)] * req_info['std_actual_rev'].loc[m] for m in park_index for n in range(N))
park_revenue_ = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in park_index for n in range(N))

char_revenue = quicksum(x_nm[(n, m)] * req_info['std_actual_rev'].loc[m] for m in charge_index for n in range(N))
char_revenue_ = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in charge_index for n in range(N))

obj1 = park_revenue + char_revenue
obj1_ = park_revenue_ + char_revenue_

# 拒绝的停车数量
refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
# 平台拒绝数量
obj4 = refuse_park + refuse_char

# 行程时间  到停车场的时间
obj2 = quicksum(
    cost_matrix[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix[z][req_info['D'].loc[m] + 2] * y_mz[
        (m, z)] for m in total_index for z in range(Z))

obj2_ = quicksum(
    cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
        (m, z)] for m in total_index for z in range(Z))

# 总目标
alpha1 = 1
alpha2 = 1
alpha4 = 5
obj = alpha1 * obj1 - alpha2 * obj2 - alpha4 * obj4

plp.setObjective(obj, GRB.MAXIMIZE)

plp.addConstrs(
    quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in total_index)
plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

plp.addConstrs(
    x_nm[(n, i)] + x_nm[(n, k)] <= 1 + Z_IK[(i, k)] + Z_IK[(k, i)] for i in range(I) for k in range(i) for n in
    range(N))

rule = 1

if rule == 1:
    # rule1
    # 停车请求只能分配到OPS
    plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
elif rule == 2:
    # rule2:
    # 停车请求可以分配到OPS和CPS
    pass
elif rule == 3:
    # 只有在规定时间内停车请求可以分配到充电桩
    # 8-10 17-19 可以停放（到达时间为480-600或1020-1140）
    filtered_df = req_info.loc[park_index]
    allocatable_index_1 = filtered_df[
        (filtered_df['actual_s'] >= 480) & (filtered_df['actual_s'] <= 600)].index.tolist()
    allocatable_index_2 = filtered_df[
        (filtered_df['actual_s'] >= 1020) & (filtered_df['actual_s'] <= 1140)].index.tolist()
    allocatable_index = allocatable_index_1 + allocatable_index_2
    # 符合条件的停车请求可以分配到OPS和CPS
    # 不符合的只能分配到OPS
    no_allocatable_index = list(set(park_index) - set(allocatable_index))
    plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)

# 快充电请求只能分配到快充CPS
plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
# 慢充电请求只能分配到慢充CPS
plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

plp.optimize()

REVENUE_total[0] = obj.getValue()
REVENUE_total[1] = obj1_.getValue()
REVENUE_total[2] = park_revenue_.getValue()
REVENUE_total[3] = char_revenue_.getValue()
REVENUE_total[4] = obj4.getValue()
REVENUE_total[5] = refuse_park.getValue()
REVENUE_total[6] = refuse_char.getValue()
REVENUE_total[7] = obj2_.getValue()

X_NM = np.zeros((N, len(total_index)))
acc_res = {n: [m for m in total_index if x_nm[(n, m)].X == 1] for n in range(N)}
for berth, user in acc_res.items():
    print(f"berth:{berth} has accepted users:{[each for each in user]}")

R_MK_Actual = get_actual_rmk(req_info)
R_MK_Submitted = get_rmk(req_info)
T_Actual = R_MK_Actual.shape[1]
T_Submitted = R_MK_Submitted.shape[1]

S_NK_Actual = np.zeros((N, T_Actual))
S_NK_Submitted = np.zeros((N, T_Submitted))

for n in range(N):
    S_NK_Actual[n] = sum(R_MK_Actual[each] for each in acc_res[n])
    S_NK_Submitted[n] = sum(R_MK_Submitted[each] for each in acc_res[n])

obj4 = quicksum(
    pl_occ(z, R_MK_Actual, EACH_INDEX[z], req_info['arrival_t'].loc[m]) * y_mz[(m, z)] for m in total_index for z in
    range(Z))

# REVENUE_total[8] = obj4.getValue()

P_obj = REVENUE_total[0]
P_rev = REVENUE_total[1]
P_park_rev = REVENUE_total[2]
P_char_rev = REVENUE_total[3]
P_refuse = REVENUE_total[4]
P_park_refuse = REVENUE_total[5]
P_char_refuse = REVENUE_total[6]
P_travel = REVENUE_total[7]

# parking utilization and charging utilization
parking_util = np.sum(R_MK_Actual[OPS_INDEX, :1440]) / (sum(pl[i].ordinary_num for i in range(Z)) * 1440)
charging_util = np.sum(R_MK_Actual[CPS_INDEX, :1440]) / (sum(pl[i].charge_num for i in range(Z)) * 1440)

# acceptance rate of reservation requests
P_acc = (I - P_refuse) / I
P_park_acc = (park_num - P_park_refuse) / park_num
P_char_acc = (park_num * charge_ratio - P_char_refuse) / (park_num * charge_ratio)

# 平均行程时间
P_travel = P_travel / (I - P_refuse)

temp_occ = np.array([np.sum(R_MK_Actual[EACH_INDEX[z]], axis=0) / pl[z].total_num for z in range(Z)]).reshape(Z,
                                                                                                              T_Actual)
P_occ_diff = np.mean(np.sum((temp_occ - np.mean(temp_occ, axis=0)) ** 2 / Z, axis=0))

result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha4": alpha4, "request number": I,
               "objective value": P_obj, "total revenue": P_rev,
               "park revenue": P_park_rev, "char revenue": P_char_rev, "refuse number": P_refuse,
               "refuse park number": P_park_refuse, "refuse char number": P_char_refuse,
               "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
               "charge util": charging_util, "travel cost": P_travel, "parking lot occ diff": P_occ_diff}

for key, value in result_dict.items():
    print(key + ": " + str(value))

fig, ax = plt.subplots(1, 2)
sns.heatmap(data=S_NK_Actual, ax=ax[0])
ax[0].set_ylabel('Actual')
sns.heatmap(data=S_NK_Submitted, ax=ax[1])
ax[1].set_ylabel('Submitted')
plt.show()

print("111")

# model check
# total_acc = [x_ij[(i, j)].X for i in range(I) for j in range(J)]
# print(sum(total_acc))
#
# # check the results
# allocation_res = {j: [i for i in range(I) if x_ij[(i, j)].X == 1] for j in range(J)}
# for berth, user in allocation_res.items():
#     print(f"berth:{berth} has accepted users:{[each for each in user]}")
#
# R_it = get_rmk(req_info).astype(int)  # 需求矩阵
# T = R_it.shape[1]  # 时长
# Q_jt = np.zeros((J, T))  # 供应矩阵 1: idle 0: occupied
#
# for j in range(J):
#     Q_jt[j] = sum(R_it[each] for each in allocation_res[j])
#
# sns.heatmap(data=Q_jt)
# plt.show()
