from gurobipy import *
from entity import OD
from strategy.utils import *

# 需求
park_num = 300
charge_ratio = 1
req_info = get_request(park_num=park_num, charge_ratio=charge_ratio)
# index
total_index, park_index, charge_index, fast_charge_index, slow_charge_index = get_index(req_info)

# basic info
O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
cost_matrix_ = OD.OdCost().cost_matrix
cost_matrix = OD.OdCost().get_std_cost()

#
N = 100

# 泊位索引
T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
_, OPS_INDEX = P_ZN(Z, N, pl)
_, CPS_INDEX = C_ZN(Z, N, pl)
_, FAST_INDEX = Fast_ZN(Z, N, pl)
_, SLOW_INDEX = Slow_ZN(Z, N, pl)



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
        sigma = np.random.uniform(low=5, high=10)
    else:
        sigma = np.random.uniform(low=15, high=30)

    # actual_s_i = np.random.normal(loc=s_i, scale=sigma)
    # actual_e_i = np.random.normal(loc=e_i, scale=sigma)
    actual_s_i = s_i - sigma
    actual_e_i = e_i + sigma

    return actual_s_i, actual_e_i


decision_interval = 15
req_info['request_interval'] = req_info['request_t'] // decision_interval
req_info['s_it'] = req_info['arrival_t'] // decision_interval
req_info['e_it'] = req_info['leave_t'] // decision_interval

request_interval = req_info['request_interval'].unique()
request_interval.sort()
I = request_interval

def get_rmk(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, max(req_info['e_it']) + 1))
    for i in range(req_num):
        start = req_info['s_it'].iloc[i]
        end = req_info['e_it'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


r_mk = get_rmk(req_info).astype(int)


def update_req_info(req_info):

    req_info[['actual_s', 'actual_e']] = req_info.apply(
        lambda x: pd.Series(actual_t(x['activity_t'], x['arrival_t'], x['leave_t'])),
        axis=1
    )
    req_info[['actual_s', 'actual_e']] = req_info[['actual_s', 'actual_e']].astype(int)
    req_info['actual_activity'] = req_info['actual_e'] - req_info['actual_s']
    # req_info[['actual_rev', 'std_actual_rev']] = pd.Series(
    #     get_revenue(req_info['actual_activity'].values, req_info['new_label'].values))
    req_info['actual_rev'] = get_revenue(req_info['actual_activity'].values, req_info['new_label'].values)
    # req_info[['rev', 'std_rev']] = pd.Series(get_revenue(req_info['activity_t'].values, req_info['new_label'].values))

    return req_info

req_info = update_req_info(req_info)

arr = req_info['actual_s']
lea = req_info['actual_e']
activity = req_info['actual_activity']

N = 100


# auxiliary variable
def z_ik(s_i, e_k, buffer=15):
    if s_i - e_k >= buffer:
        return 1
    else:
        return 0


rule = 1
total_acc = 0
park_rev = 0
char_rev = 0
park_re = 0
char_re = 0
total_re = 0
travel_t = 0

for ith, i in enumerate(I):

    # 定义一个已经分配的索引集合
    assign_result = {}
    #
    alpha1 = 1
    alpha2 = 5
    alpha3 = 1

    #
    temp_req_info = req_info[req_info['request_interval'] == i]
    total_index = temp_req_info.index.tolist()
    park_index = temp_req_info[temp_req_info['label'] == 0].index.tolist()
    charge_index = temp_req_info[temp_req_info['label'] == 1].index.tolist()
    fast_charge_index = temp_req_info[temp_req_info['new_label'] == 1].index.tolist()
    slow_charge_index = temp_req_info[temp_req_info['new_label'] == 2].index.tolist()

    if ith == 0:

        plp = Model("linearized model")

        x_nm = plp.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_mz = plp.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z
        z_mm = plp.addVars({(m, m_) for m in total_index for m_ in total_index if m != m_}, vtype=GRB.BINARY,
                           name="z_mm")

        Z_MM = {(m, m_): z_ik(arr[m], lea[m_]) for m in total_index for m_ in total_index if m != m_}

        park_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in park_index for n in range(N))
        char_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in charge_index for n in range(N))
        obj1 = park_revenue + char_revenue

        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
        obj2 = refuse_park + refuse_char

        obj3 = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in total_index for z in range(Z))

        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3

        plp.setObjective(obj, GRB.MAXIMIZE)

        plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
        plp.addConstrs(
            quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
            total_index)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            OPS_INDEX for m in park_index for m_ in park_index if m != m_)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            FAST_INDEX for m in fast_charge_index for m_ in fast_charge_index if m != m_)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            SLOW_INDEX for m in slow_charge_index for m_ in slow_charge_index if m != m_)

        if rule == 1:
            # rule1
            # 停车请求只能分配到OPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
        elif rule == 2:
            # rule2:
            # 停车请求可以分配到OPS和CPS
            pass

        # 快充电请求只能分配到快充CPS
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
        # 慢充电请求只能分配到慢充CPS
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        plp.optimize()

        park_rev += park_revenue.getValue()
        char_rev += char_revenue.getValue()
        park_re += refuse_park.getValue()
        char_re += refuse_char.getValue()
        total_re += obj3.getValue() / alpha2
        travel_t += obj2.getValue()

        for m in total_index:
            for n in range(N):
                if x_nm[(n, m)].X == 1:
                    assign_result[m] = n
                else:
                    break

        results = {
            "Assignments": [],
            "Schedule": [],
            "Revenue": park_re + char_re
        }
        if plp.status == GRB.OPTIMAL:
            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].x > 0.5:  # Binary variable, check if assigned
                        results["Assignments"].append((m, n))
            for (i, j) in Z_MM.keys():
                if z_mm[i, j].x > 0.5:
                    results["Schedule"].append((i, "before", j))

        print(results)

    # 第一次分配之后 要维护和更新ZMM
    else:
        """
        计算Z_MM矩阵,此时矩阵的元素为已分配的用户（视为virtual owner）和新一轮中新发出的请求
        """

        plp = Model("linearized model")

        previous_assigned = [m for m in assign_result.keys() if req_info['e_it'].loc[m] >= i]
        previous_assigned_park = [m for m in previous_assigned if req_info['new_label'].loc[m] == 0]
        previous_assigned_fc = [m for m in previous_assigned if req_info['new_label'].loc[m] == 1]
        previous_assigned_sc = [m for m in previous_assigned if req_info['new_label'].loc[m] == 2]

        t_total_index = total_index + previous_assigned
        t_park_index = park_index + previous_assigned_park
        t_charge_index = charge_index + previous_assigned_fc + previous_assigned_sc
        t_fast_charge_index = fast_charge_index + previous_assigned_fc
        t_slow_charge_index = slow_charge_index + previous_assigned_sc

        x_nm = plp.addVars({(n, m) for n in range(N) for m in t_total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_mz = plp.addVars({(m, z) for m in t_total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z
        z_mm = plp.addVars({(m, m_) for m in t_total_index for m_ in t_total_index if m != m_}, vtype=GRB.BINARY,
                           name="z_mm")

        Z_MM = {(m, m_): z_ik(arr[m], lea[m_]) for m in t_total_index for m_ in t_total_index if m != m_}

        park_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in park_index for n in range(N))
        char_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in charge_index for n in range(N))
        obj1 = park_revenue + char_revenue

        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
        obj2 = refuse_park + refuse_char

        obj3 = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in total_index for z in range(Z))

        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3

        plp.setObjective(obj, GRB.MAXIMIZE)

        plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
        plp.addConstrs(
            quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
            total_index)

        plp.addConstrs(
            x_nm[(assign_result[assigned_driver], assigned_driver)] == 1 for assigned_driver in previous_assigned)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            OPS_INDEX for m in t_park_index for m_ in t_park_index if m != m_)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            FAST_INDEX for m in t_fast_charge_index for m_ in t_fast_charge_index if m != m_)

        plp.addConstrs(
            x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n in
            SLOW_INDEX for m in t_slow_charge_index for m_ in t_slow_charge_index if m != m_)

        if rule == 1:
            # rule1
            # 停车请求只能分配到OPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
        elif rule == 2:
            # rule2:
            # 停车请求可以分配到OPS和CPS
            pass

        # 快充电请求只能分配到快充CPS
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
        # 慢充电请求只能分配到慢充CPS
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
        plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        plp.optimize()

        park_rev += park_revenue.getValue()
        char_rev += char_revenue.getValue()
        park_re += refuse_park.getValue()
        char_re += refuse_char.getValue()
        total_re += obj3.getValue() / alpha2
        travel_t += obj2.getValue()

        for m in total_index:
            for n in range(N):
                if x_nm[(n, m)].X == 1:
                    assign_result[m] = n
                else:
                    break

        results = {
            "Assignments": [],
            "Schedule": [],
            "Revenue": park_re + char_re
        }
        if plp.status == GRB.OPTIMAL:
            for m in t_total_index:
                for n in range(N):
                    if x_nm[(n, m)].x > 0.5:  # Binary variable, check if assigned
                        results["Assignments"].append((m, n))
            for (i, j) in Z_MM.keys():
                if z_mm[i, j].x > 0.5:
                    results["Schedule"].append((i, "before", j))

        print(results)

print(park_rev)
print(char_rev)
print(park_re)
print(char_re)
print(total_re)
print(travel_t)