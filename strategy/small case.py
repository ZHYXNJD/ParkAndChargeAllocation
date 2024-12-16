import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from entity import OD
from utils import *
from gurobipy import *
import pandas as pd
from collections import defaultdict

np.random.seed(43)


def get_rmk(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, K))
    for i in range(req_num):
        start = req_info['arrival_interval'].iloc[i]
        end = req_info['leave_interval'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


def get_rmk_a(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, K))
    for i in range(req_num):
        start = req_info['actual_s_it'].iloc[i]
        end = req_info['actual_e_it'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


# calculate user's actual arrival and departure time
# this can be learned from history records
# for simple we assume the distribution if independent of arrival and departure time
# for short time activity,the arrival and departure unpunctuality sigma follows the uniform distribution of 0-6 min
# for long time activity,the arrival and departure unpunctuality sigma follows the uniform distribution of 5-25 min
def actual_t(s_i, e_i):
    """
    s_i: submitted start time
    e_i: submitted end time
    """
    sigma = np.random.uniform(low=3, high=15)
    # if activity_t <= 180:
    #     sigma = np.random.uniform(low=5, high=10)
    # else:
    #     sigma = np.random.uniform(low=10, high=15)
    #
    actual_s_i = np.random.normal(loc=s_i, scale=sigma)
    actual_e_i = np.random.normal(loc=e_i, scale=sigma)

    # actual_s_i = s_i - sigma
    # actual_e_i = e_i + sigma

    return actual_s_i, actual_e_i


def update_req_info(req_info):
    # req_info[['actual_s', 'actual_e']] = req_info.apply(
    #     lambda x: pd.Series(actual_t(x['activity_t'], x['arrival_t'], x['leave_t'])),
    #     axis=1
    # )

    req_info[['actual_s', 'actual_e']] = req_info.apply(
        lambda x: pd.Series(actual_t(x['arrival_t'], x['leave_t'])),
        axis=1
    )

    req_info[['actual_s', 'actual_e']] = req_info[['actual_s', 'actual_e']].astype(int)
    req_info['actual_s_it'] = req_info['actual_s'] // decision_interval
    req_info['actual_e_it'] = req_info['actual_e'] // decision_interval

    req_info['actual_activity'] = req_info['actual_e'] - req_info['actual_s']
    req_info['actual_rev'] = get_revenue(req_info['actual_activity'].values, req_info['new_label'].values)
    req_info['revenue'] = get_revenue(req_info['activity_t'].values, req_info['new_label'].values)

    return req_info


# auxiliary variable
def z_ik(s_k, e_i, buffer):
    if s_k - e_i >= buffer:
        return 1
    else:
        return 0


def dp(rule):
    """
    alpha1:收益系数
    alpha2:拒绝系数
    alpha3:时间系数
    """
    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []
    # 记录分配结果
    res = [0, 0, 0, 0, 0, 0, 0, 0]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；
    X_NM = defaultdict(int)
    Y_ZM = defaultdict(int)

    for ith, i in enumerate(request_interval):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        model = Model("dp")

        temp_req_info = req_info[req_info['request_interval'] == i]
        total_index = temp_req_info.index.tolist()
        park_index = temp_req_info[temp_req_info['label'] == 0].index.tolist()
        charge_index = temp_req_info[temp_req_info['label'] == 1].index.tolist()
        fast_charge_index = temp_req_info[temp_req_info['new_label'] == 1].index.tolist()
        slow_charge_index = temp_req_info[temp_req_info['new_label'] == 2].index.tolist()

        x_nm = model.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_mz = model.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        model.update()

        # 平台收益：预约费 停车费 充电费
        park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
        char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))
        obj1 = park_revenue + char_revenue

        # 拒绝数量
        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)
        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)
        obj2 = refuse_park + refuse_char

        park_travel = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in park_index for z in range(Z))

        charge_travel = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in charge_index for z in range(Z))

        obj3 = park_travel + charge_travel

        # 总目标
        # obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
        # model.setObjective(obj, GRB.MAXIMIZE)
        model.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0, priority=5)
        model.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1, priority=3)

        model.addConstrs(quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= S_NK[n][k] for n in
                         range(N) for k in range(K))

        model.addConstrs(
            quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
            total_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

        if rule == 0:
            # rule0
            # 停车请求只能分配到OPS
            model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
        elif rule == 1:
            pass
        elif rule == 2:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
        # 6-10 14:18
        elif rule == 3:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index_1 = filtered_df[
                (filtered_df['arrival_t'] < 360) | (filtered_df['arrival_t'] > 1020)].index.tolist()
            no_allocatable_index_2 = filtered_df[
                (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
            model.addConstrs(
                quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index_1 + no_allocatable_index_2)
        # 短时停车放快充 长时停车放慢充
        else:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_1)
            no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_2)

        # 快充电请求只能分配到快充CPS
        model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

        # 慢充电请求只能分配到慢充CPS
        model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        model.optimize()

        res[1] += obj1.getValue()  # 停车场收益
        res[2] += park_revenue.getValue()  # 停车收益
        res[3] += char_revenue.getValue()  # 充电收益
        res[4] += obj2.getValue()  # 拒绝总数量
        res[5] += refuse_park.getValue()  # 停车拒绝数
        res[6] += refuse_char.getValue()  # 充电拒绝数
        res[7] += obj3.getValue()  # 行程时间

        assign_index = []

        for m in total_index:
            for n in range(N):
                if x_nm[(n, m)].X == 1:
                    S_NK[n] -= r_mk[m]
                    S_NK_A[n] -= r_mk_a[m]
                    X_NM[(n, m)] = 1
                    assign_index.append(m)
                    for z in range(Z):
                        if y_mz[(m, z)].X == 1:
                            Y_ZM[(z, m)] = 1
                            assign_info.append([m, n, z, i])
                    break

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :96]) / (sum(pl[l].ordinary_num for l in range(Z)) * 96)
    charging_util = np.sum(S_NK[CPS_INDEX, :96]) / (sum(pl[l].charge_num for l in range(Z)) * 96)
    total_util = np.sum(S_NK[:, :96]) / (sum(pl[l].total_num for l in range(Z)) * 96)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    acc_num = req_num - res[4]
    P_acc = acc_num / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (charge_num - res[6]) / charge_num
    # travel cost
    res[7] /= acc_num

    # 成功率
    faile_num = len(np.where(S_NK_A < 0)[0])
    success_park = (acc_num - faile_num) / acc_num

    # 冲突惩罚
    confilct_cost = 5 * faile_num
    # 净收益
    net_revenue = res[1] - confilct_cost

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "total_util": total_util, "travel cost": res[-1],
                   "success ratio": success_park,
                   "conflict cost": confilct_cost, 'net revenue': net_revenue}

    print("-------dp-------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    # fig, axes = plt.subplots(1, 2)
    # sns.heatmap(S_NK[:, 31:47], ax=axes[0])
    # sns.heatmap(S_NK_A[:, 31:47], ax=axes[1])
    # axes[0].set_title('Submitted')
    # axes[1].set_title('Actual')
    # # plt.title(f"unp-dp:{rule}")
    # plt.show()

    # sns.heatmap(S_NK)
    # plt.title(f"dp:{rule}")
    # plt.show()

    if need_save:

        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\save_data_1121')

        try:
            os.makedirs(f'{park_arrival_num}-1')
        except:
            pass
        os.chdir(f'{park_arrival_num}-1')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'SNK_info', 'SNKA_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/dp_{rule}.npy", S_NK)
        np.save(f"SNKA_info/adp_{rule}.npy", S_NK_A)
        np.save(f"result_info/dp_{rule}.npy", result_dict)
        assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                        data=np.array(assign_info).reshape((-1, 4)))
        merge_data = pd.merge(assign_info_data, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/dp_{rule}.csv", index=False)


def dp_unpunctuality(rule):
    # arr = req_info['arrival_interval']
    # lea = req_info['leave_interval']

    arr = req_info['arrival_t']
    lea = req_info['leave_t']

    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []
    # 记录分配结果
    X_NM = {}
    Y_ZM = {}

    for ith, i in enumerate(request_interval):

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

            Z_MM = {(m, m_): z_ik(arr[m_], lea[m]) for m in total_index for m_ in total_index if m != m_}

            park_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in park_index for n in range(N))
            char_revenue = quicksum(
                x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in charge_index for n in range(N))
            obj1 = park_revenue + char_revenue

            refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
            refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
            obj2 = refuse_park + refuse_char

            park_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in park_index for z in range(Z))

            charge_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in charge_index for z in range(Z))

            obj3 = park_travel + charge_travel

            plp.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0,
                              priority=5)
            plp.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1,
                              priority=3)

            plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
            plp.addConstrs(
                quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
                total_index)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
                in
                OPS_INDEX for m in park_index for m_ in park_index if m != m_)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
                in
                FAST_INDEX for m in fast_charge_index for m_ in fast_charge_index if m != m_)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
                in
                SLOW_INDEX for m in slow_charge_index for m_ in slow_charge_index if m != m_)

            if rule == 0:
                # rule0
                # 停车请求只能分配到OPS
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
            elif rule == 1:
                pass
            elif rule == 2:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            # 6-10 14:18
            elif rule == 3:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[
                    (filtered_df['arrival_t'] < 360) & (filtered_df['arrival_t'] > 1020)].index.tolist()
                no_allocatable_index_2 = filtered_df[
                    (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
                plp.addConstrs(
                    quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in
                    no_allocatable_index_1 + no_allocatable_index_2)
            # 短时停车放快充 长时停车放慢充
            else:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_1)
                no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_2)

            # 快充电请求只能分配到快充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

            # 慢充电请求只能分配到慢充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

            plp.optimize()

            res[1] += obj1.getValue()  # 停车场收益
            res[2] += park_revenue.getValue()  # 停车收益
            res[3] += char_revenue.getValue()  # 充电收益
            res[4] += obj2.getValue()  # 拒绝总数量
            res[5] += refuse_park.getValue()  # 停车拒绝数
            res[6] += refuse_char.getValue()  # 充电拒绝数
            res[7] += obj3.getValue()  # 行程时间

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[m] = z
                                assign_info.append([m, n, z, i])
                        break

        else:
            """
            计算Z_MM矩阵,此时矩阵的元素为已分配的用户（视为virtual owner）和新一轮中新发出的请求
            """
            plp = Model("linearized model")

            previous_assigned = [m for m in X_NM.keys() if req_info['leave_interval'].loc[m] >= i]
            t_total_index = total_index + previous_assigned

            x_nm = plp.addVars({(n, m) for n in range(N) for m in t_total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
            y_mz = plp.addVars({(m, z) for m in t_total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z
            z_mm = plp.addVars({(m, m_) for m in t_total_index for m_ in t_total_index if m != m_}, vtype=GRB.BINARY,
                               name="z_mm")

            Z_MM = {(m, m_): z_ik(arr[m_], lea[m]) for m in t_total_index for m_ in t_total_index if m != m_}

            park_revenue = quicksum(x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in park_index for n in range(N))
            char_revenue = quicksum(
                x_nm[(n, m)] * req_info['actual_rev'].loc[m] for m in charge_index for n in range(N))
            obj1 = park_revenue + char_revenue

            refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
            refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))
            obj2 = refuse_park + refuse_char

            park_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in park_index for z in range(Z))

            charge_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in charge_index for z in range(Z))

            obj3 = park_travel + charge_travel

            plp.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0,
                              priority=5)
            plp.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1,
                              priority=3)

            plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
            plp.addConstrs(
                quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
                total_index)

            plp.addConstrs(
                x_nm[(X_NM[assigned_driver], assigned_driver)] == 1 for assigned_driver in previous_assigned)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
                in range(N) for m in t_total_index for m_ in t_total_index if m != m_)

            if rule == 0:
                # rule0
                # 停车请求只能分配到OPS
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
            elif rule == 1:
                pass
            elif rule == 2:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            # 6-10 14:18
            elif rule == 3:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[
                    (filtered_df['arrival_t'] < 360) & (filtered_df['arrival_t'] > 1020)].index.tolist()
                no_allocatable_index_2 = filtered_df[
                    (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
                plp.addConstrs(
                    quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in
                    no_allocatable_index_1 + no_allocatable_index_2)
            # 短时停车放快充 长时停车放慢充
            else:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_1)
                no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_2)
                pass
            # 快充电请求只能分配到快充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

            # 慢充电请求只能分配到慢充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

            plp.optimize()

            res[1] += obj1.getValue()  # 停车场收益
            res[2] += park_revenue.getValue()  # 停车收益
            res[3] += char_revenue.getValue()  # 充电收益
            res[4] += obj2.getValue()  # 拒绝总数量
            res[5] += refuse_park.getValue()  # 停车拒绝数
            res[6] += refuse_char.getValue()  # 充电拒绝数
            res[7] += obj3.getValue()  # 行程时间

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[(z, m)] = 1
                                assign_info.append([m, n, z, i])
                        break

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX][:, :48]) / (sum(pl[l].ordinary_num for l in range(Z)) * 48)
    charging_util = np.sum(S_NK[CPS_INDEX][:, :48]) / (sum(pl[l].charge_num for l in range(Z)) * 48)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    P_acc = (req_num - res[4]) / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (park_arrival_num * charge_ratio - res[6]) / (park_arrival_num * charge_ratio)

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "travel cost": res[7] / (req_num - res[4])}

    print("-------unpun-dp-------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    fig, axes = plt.subplots(1, 2)
    sns.heatmap(S_NK, ax=axes[0])
    sns.heatmap(S_NK_A, ax=axes[1])
    axes[0].set_title('Submitted')
    axes[1].set_title('Actual')
    # plt.title(f"unp-dp:{rule}")
    plt.show()

    if need_save:

        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\test_case_1119')

        try:
            os.makedirs(f'{park_arrival_num}-{charge_ratio}')
        except:
            pass
        os.chdir(f'{park_arrival_num}-{charge_ratio}')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'revenue_info', 'SNK_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/unp-dp_{rule}.npy", S_NK)
        np.save(f"revenue_info/unp-dp_{rule}.npy", res)
        np.save(f"result_info/unp-dp_{rule}.npy", result_dict)
        assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                        data=np.array(assign_info).reshape((-1, 4)))
        merge_data = pd.merge(assign_info_data, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/unp-dp_{rule}.csv", index=False)


def rdp(rule, ops_num):
    """
    alpha1:收益系数
    alpha2:拒绝系数
    alpha3:时间系数
    """

    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []
    # 记录分配结果
    res = [0, 0, 0, 0, 0, 0, 0, 0]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；
    X_NM = {}
    Y_ZM = {}

    # 第一轮分配不用调整
    need_adjust = False

    # 存储上一轮已经分配完成且可以调整的请求索引
    adj_total_index = []
    adj_park_index = []
    adj_charge_index = []
    adj_fast_charge_index = []
    adj_slow_charge_index = []
    for ith, i in enumerate(I):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        model = Model("rdp")

        # 本轮的请求
        this_req_info = req_info[req_info['request_interval'] == i]
        total_index = this_req_info.index.tolist()
        park_index = this_req_info[this_req_info['label'] == 0].index.tolist()
        charge_index = this_req_info[this_req_info['label'] == 1].index.tolist()
        fast_charge_index = this_req_info[this_req_info['new_label'] == 1].index.tolist()
        slow_charge_index = this_req_info[this_req_info['new_label'] == 2].index.tolist()

        x_nm = model.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_mz = model.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        if need_adjust:
            x_na = model.addVars({(n, a) for n in range(N) for a in adj_total_index}, vtype=GRB.BINARY,
                                 name="x_na")  # n*m
            y_az = model.addVars({(a, z) for a in adj_total_index for z in range(Z)}, vtype=GRB.BINARY,
                                 name="y_az")  # m*z

        model.update()

        # 平台收益：预约费 停车费 充电费
        park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
        char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))
        obj1 = park_revenue + char_revenue

        # 拒绝数量
        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)
        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)
        obj2 = refuse_park + refuse_char

        park_travel = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in park_index for z in range(Z)) + \
                      need_adjust * quicksum(
            cost_matrix_[z][req_info['O'].loc[a]] * y_az[(a, z)] + 2 * cost_matrix_[z][req_info['D'].loc[a] + 2] * y_az[
                (a, z)] for a in adj_park_index for z in range(Z))

        charge_travel = quicksum(
            cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in charge_index for z in range(Z)) + \
                        need_adjust * quicksum(
            cost_matrix_[z][req_info['O'].loc[a]] * y_az[(a, z)] + 2 * cost_matrix_[z][req_info['D'].loc[a] + 2] * y_az[
                (a, z)] for a in adj_charge_index for z in range(Z))

        obj3 = park_travel + charge_travel

        # 总目标
        # obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
        # model.setObjective(obj, GRB.MAXIMIZE)
        model.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0, priority=5)
        model.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1, priority=3)

        if need_adjust:

            model.addConstrs(quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) + quicksum(
                x_na[(n, a)] * r_mk[a][k] for a in adj_total_index) <= S_NK[n][k] for n in range(N) for k in
                             range(K))

            model.addConstrs(
                quicksum(T_ZN[z][n] * x_na[(n, a)] for n in range(N)) == y_az[(a, z)] for z in range(Z) for a in
                adj_total_index)

            model.addConstrs(quicksum(x_na[(n, a)] for n in range(N)) == 1 for a in adj_total_index)

        else:
            model.addConstrs(quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= S_NK[n][k] for n in
                             range(N) for k in range(K))

        model.addConstrs(
            quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
            total_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

        if rule == 0:
            # rule0
            # 停车请求只能分配到OPS
            model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
            model.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in adj_park_index)
        elif rule == 1:
            pass
        elif rule == 2:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            adj_filtered_df = req_info.loc[adj_park_index]
            adj_no_allocatable_index = adj_filtered_df[adj_filtered_df['activity_t'] >= 450].index.tolist()
            model.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in adj_no_allocatable_index)
        # 6-10 14:18
        elif rule == 3:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index_1 = filtered_df[
                (filtered_df['arrival_t'] < 360) | (filtered_df['arrival_t'] > 1020)].index.tolist()
            no_allocatable_index_2 = filtered_df[
                (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
            model.addConstrs(
                quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index_1 + no_allocatable_index_2)

            adj_filtered_df = req_info.loc[adj_park_index]
            no_adj_allocatable_index_1 = adj_filtered_df[
                (adj_filtered_df['arrival_t'] < 360) | (adj_filtered_df['arrival_t'] > 1020)].index.tolist()
            no_adj_allocatable_index_2 = adj_filtered_df[
                (adj_filtered_df['arrival_t'] > 540) & (adj_filtered_df['arrival_t'] < 840)].index.tolist()
            model.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in
                             no_adj_allocatable_index_1 + no_adj_allocatable_index_2)
        # 短时停车放快充 长时停车放慢充
        else:
            filtered_df = req_info.loc[park_index]
            no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= sl_cut].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_1)
            no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > sl_cut].index.tolist()
            model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_2)

            adj_filtered_df = req_info.loc[adj_park_index]
            adj_no_allocatable_index_1 = adj_filtered_df[adj_filtered_df['activity_t'] <= sl_cut].index.tolist()
            model.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in adj_no_allocatable_index_1)
            adj_no_allocatable_index_2 = adj_filtered_df[adj_filtered_df['activity_t'] > sl_cut].index.tolist()
            model.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in adj_no_allocatable_index_2)

        # 快充电请求只能分配到快充CPS
        model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

        model.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in adj_fast_charge_index)
        model.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_fast_charge_index)

        # 慢充电请求只能分配到慢充CPS
        model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
        model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        model.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in adj_slow_charge_index)
        model.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_slow_charge_index)

        model.optimize()

        res[1] += obj1.getValue()  # 停车场收益
        res[2] += park_revenue.getValue()  # 停车收益
        res[3] += char_revenue.getValue()  # 充电收益
        res[4] += obj2.getValue()  # 拒绝总数量
        res[5] += refuse_park.getValue()  # 停车拒绝数
        res[6] += refuse_char.getValue()  # 充电拒绝数
        res[7] += obj3.getValue()  # 行程时间

        if need_adjust:

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[m] = z
                                assign_info.append([m, n, z, i])
                        break

            for a in adj_total_index:
                for n in range(N):
                    if x_na[(n, a)].X == 1:
                        S_NK[n] -= r_mk[a]
                        S_NK_A[n] -= r_mk_a[a]
                        X_NM[a] = n
                        assign_index.append(a)
                        for z in range(Z):
                            if y_az[(a, z)].X == 1:
                                Y_ZM[a] = z
                                assign_info.append([a, n, z, i])
                        break

            # 存储分配的信息
            # 将上一轮可以调整的请求和本轮可以调整的请求合并 继续调整
            # 本轮可以调整的请求索引
            this_adj_total_index = list(
                set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            this_adj_park_index = list(set(this_adj_total_index) & set(park_index))
            this_adj_charge_index = list(set(this_adj_total_index) & set(charge_index))
            this_adj_fast_charge_index = list(set(this_adj_charge_index) & set(fast_charge_index))
            this_adj_slow_charge_index = list(set(this_adj_charge_index) & set(slow_charge_index))

            # 上一轮可以继续调整的请求索引
            req_info.loc[adj_total_index, 'diff'] = req_info.loc[adj_total_index, 'diff'] - 1
            secondary_adj_total_index = [idx for idx in adj_total_index if req_info.at[idx, 'diff'] > 1]
            secondary_adj_park_index = list(set(secondary_adj_total_index) & set(adj_park_index))
            secondary_adj_charge_index = list(set(secondary_adj_total_index) & set(adj_charge_index))
            secondary_adj_fast_charge_index = list(set(secondary_adj_charge_index) & set(adj_fast_charge_index))
            secondary_adj_slow_charge_index = list(set(secondary_adj_charge_index) & set(adj_slow_charge_index))

            # 合并后作为总的可以调整的索引
            adj_total_index = secondary_adj_total_index + this_adj_total_index
            adj_park_index = secondary_adj_park_index + this_adj_park_index
            adj_charge_index = secondary_adj_charge_index + this_adj_charge_index
            adj_fast_charge_index = secondary_adj_fast_charge_index + this_adj_fast_charge_index
            adj_slow_charge_index = secondary_adj_slow_charge_index + this_adj_slow_charge_index

            # 恢复SNK
            for adj_index in adj_total_index:
                space_n = X_NM[adj_index]
                S_NK[space_n] += r_mk[adj_index]
                S_NK_A[space_n] += r_mk_a[adj_index]
                del X_NM[adj_index]


        else:

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[m] = z
                                assign_info.append([m, n, z, i])
                        break

            # 本轮可以调整的请求索引
            adj_total_index = list(set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            adj_park_index = list(set(adj_total_index) & set(park_index))
            adj_charge_index = list(set(adj_total_index) & set(charge_index))
            adj_fast_charge_index = list(set(adj_charge_index) & set(fast_charge_index))
            adj_slow_charge_index = list(set(adj_charge_index) & set(slow_charge_index))

            # 恢复SNK
            for adj_index in adj_total_index:
                space_n = X_NM[adj_index]
                S_NK[space_n] += r_mk[adj_index]
                S_NK_A[space_n] += r_mk_a[adj_index]
                del X_NM[adj_index]

        need_adjust = True

    # 计算收益
    # 对多次分配的请求删除合并，计算收益
    dpr_assign = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                              data=np.array(assign_info).reshape((-1, 4)))
    dpr_assign_filter = dpr_assign.loc[dpr_assign.groupby('req_id')['assign_t'].idxmax()]
    dpr_assign_filter.reset_index(drop=True, inplace=True)
    travel_cost = []
    for i in range(len(dpr_assign_filter)):
        temp_o = req_info['O'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_d = req_info['D'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_pl = dpr_assign_filter['pl_num'].iloc[i]
        travel_cost.append(cost_matrix_[int(temp_pl)][int(temp_o)] + 2 * cost_matrix_[int(temp_pl)][int(temp_d) + 2])
    dpr_assign_filter['travel_cost'] = travel_cost
    average_travel_cost = dpr_assign_filter['travel_cost'].sum() / len(dpr_assign_filter)
    res[-1] = average_travel_cost

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :96]) / (sum(pl[l].ordinary_num for l in range(Z)) * 96)
    charging_util = np.sum(S_NK[CPS_INDEX, :96]) / (sum(pl[l].charge_num for l in range(Z)) * 96)
    total_util = np.sum(S_NK[:, :96]) / (sum(pl[l].total_num for l in range(Z)) * 96)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    acc_num = req_num - res[4]
    P_acc = acc_num / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (charge_num - res[6]) / charge_num

    # 成功率
    faile_num = len(np.where(S_NK_A < 0)[0])
    success_park = (acc_num - faile_num) / acc_num

    # 冲突惩罚
    confilct_cost = 5 * faile_num
    # 净收益
    net_revenue = res[1] - confilct_cost

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "total util": total_util, "travel cost": res[-1],
                   "success ratio": success_park,
                   "conflict cost": confilct_cost, 'net revenue': net_revenue}

    print("-------rdp------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    # fig, axes = plt.subplots(1, 2)
    # vmin_ = min(S_NK.min(), S_NK_A.min())
    # sns.heatmap(S_NK[:, 31:47], ax=axes[0], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # sns.heatmap(S_NK_A[:, 31:47], ax=axes[1], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # axes[0].set_title('submitted request', fontsize=14)
    # axes[1].set_title('actual request', fontsize=14)
    # axes[1].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[0].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[1].set_xticks([i for i in range(0, 18, 2)], )
    # axes[0].set_xticks([i for i in range(0, 18, 2)])
    # axes[1].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # axes[0].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # # axes[0].set_xlabel('Decision interval', fontsize=14)
    # # axes[1].set_xlabel('Decision interval', fontsize=14)
    # axes[0].set_ylabel('parking space', fontsize=14)
    # # axes[1].set_ylabel('Parking space', fontsize=16)
    # plt.tight_layout()
    # # plt.savefig('fbfs_test.png', dpi=300)
    # plt.show()

    if need_save:

        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\demand_and_supply_analysis')

        try:
            os.makedirs(f'{park_arrival_num}-{charge_num}')
        except:
            pass
        os.chdir(f'{park_arrival_num}-{charge_num}')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'SNK_info', 'SNKA_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/rdp_{rule}_{ops_num}.npy", S_NK)
        np.save(f"SNKA_info/ardp_{rule}{ops_num}.npy", S_NK_A)
        np.save(f"result_info/rdp_{rule}{ops_num}.npy", result_dict)
        merge_data = pd.merge(dpr_assign_filter, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/rdp_{rule}{ops_num}.csv", index=False)

        # fig, axes = plt.subplots(1, 2)
        # vmin_ = min(S_NK.min(), S_NK_A.min())
        # sns.heatmap(S_NK[:, 31:47], ax=axes[0], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
        #             vmin=vmin_, vmax=1, cbar=False)
        # sns.heatmap(S_NK_A[:, 31:47], ax=axes[1], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
        #             vmin=vmin_, vmax=1, cbar=False)
        # axes[0].set_title('submitted request', fontsize=16)
        # axes[1].set_title('actual request', fontsize=16)
        # axes[1].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
        # axes[0].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
        # axes[1].set_xticks([i for i in range(0, 18, 2)], )
        # axes[0].set_xticks([i for i in range(0, 18, 2)])
        # axes[1].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
        # axes[0].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
        # axes[0].set_xlabel('Decision interval', fontsize=16)
        # axes[1].set_xlabel('Decision interval', fontsize=16)
        # axes[0].set_ylabel('Parking space', fontsize=16)
        # axes[1].set_ylabel('Parking space', fontsize=16)
        # plt.tight_layout()
        # plt.savefig('rdp.png', dpi=300)
        # plt.show()


def rdp_unpunctuality(rule, version, buffer):
    """
    alpha1:收益系数
    alpha2:拒绝系数
    alpha3:时间系数
    """

    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []

    # arr = req_info['arrival_interval']
    # lea = req_info['leave_interval']

    arr = req_info['arrival_t']
    lea = req_info['leave_t']

    # 记录分配结果
    res = [0, 0, 0, 0, 0, 0, 0, 0]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；
    X_NM = {}
    Y_ZM = {}
    Z_MM = {}

    # 存储上一轮已经分配完成且可以调整的请求索引
    adj_total_index = []
    adj_park_index = []
    adj_charge_index = []
    adj_fast_charge_index = []
    adj_slow_charge_index = []
    for ith, i in enumerate(I):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        # 本轮的请求
        this_req_info = req_info[req_info['request_interval'] == i]
        total_index = this_req_info.index.tolist()
        park_index = this_req_info[this_req_info['label'] == 0].index.tolist()
        charge_index = this_req_info[this_req_info['label'] == 1].index.tolist()
        fast_charge_index = this_req_info[this_req_info['new_label'] == 1].index.tolist()
        slow_charge_index = this_req_info[this_req_info['new_label'] == 2].index.tolist()

        if ith == 0:

            plp = Model("linearized model")

            x_nm = plp.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
            y_mz = plp.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z
            # z_mm = plp.addVars({(m, m_) for m in total_index for m_ in total_index if m != m_}, vtype=GRB.BINARY,
            #                    name="z_mm")

            for m in total_index:
                for m_ in total_index:
                    if m != m_:
                        Z_MM[(m, m_)] = z_ik(arr[m_], lea[m], buffer)

            plp.update()

            # 平台收益：预约费 停车费 充电费
            park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
            char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))
            obj1 = park_revenue + char_revenue

            # 拒绝数量
            refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)
            refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)
            obj2 = refuse_park + refuse_char

            park_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in park_index for z in range(Z))

            charge_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in charge_index for z in range(Z))

            obj3 = park_travel + charge_travel

            # 总目标
            # obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
            # model.setObjective(obj, GRB.MAXIMIZE)
            plp.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0,
                              priority=5)
            plp.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1,
                              priority=3)

            plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
            plp.addConstrs(
                quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
                total_index)
            plp.addConstrs(
                quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= 1 for n in range(N) for k in range(K))

            # plp.addConstrs(
            #     x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
            #     in range(N) for m in total_index for m_ in total_index if m != m_)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + Z_MM[(m, m_)] + Z_MM[(m_, m)] for n
                in range(N) for m in total_index for m_ in total_index if m != m_)

            if rule == 0:
                # rule0
                # 停车请求只能分配到OPS
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
            elif rule == 1:
                pass
            elif rule == 2:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            # 6-10 14:18
            elif rule == 3:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[
                    (filtered_df['arrival_t'] < 360) | (filtered_df['arrival_t'] > 1020)].index.tolist()
                no_allocatable_index_2 = filtered_df[
                    (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
                plp.addConstrs(
                    quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in
                    no_allocatable_index_1 + no_allocatable_index_2)
            # 短时停车放慢充 长时停车放快充
            else:
                filtered_df = req_info.loc[park_index]
                no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_1)
                no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_2)

            # 快充电请求只能分配到快充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

            # 慢充电请求只能分配到慢充CPS
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
            plp.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

            plp.optimize()

            res[1] += obj1.getValue()  # 停车场收益
            res[2] += park_revenue.getValue()  # 停车收益
            res[3] += char_revenue.getValue()  # 充电收益
            res[4] += obj2.getValue()  # 拒绝总数量
            res[5] += refuse_park.getValue()  # 停车拒绝数
            res[6] += refuse_char.getValue()  # 充电拒绝数
            res[7] += obj3.getValue()  # 行程时间

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[m] = z
                                assign_info.append([m, n, z, i])
                        break

            # 本轮可以调整的请求索引
            adj_total_index = list(set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            adj_park_index = list(set(adj_total_index) & set(park_index))
            adj_charge_index = list(set(adj_total_index) & set(charge_index))
            adj_fast_charge_index = list(set(adj_charge_index) & set(fast_charge_index))
            adj_slow_charge_index = list(set(adj_charge_index) & set(slow_charge_index))

            # 恢复SNK
            for adj_index in adj_total_index:
                space_n = X_NM[adj_index]
                S_NK[space_n] += r_mk[adj_index]
                S_NK_A[space_n] += r_mk_a[adj_index]
                del X_NM[adj_index]

        else:
            """
            计算Z_MM矩阵,此时矩阵的元素为已分配的用户（视为virtual owner）和新一轮中新发出的请求
            每个泊位上只保留最后占用泊位的用户
            """
            previous_assigned = [m for m in X_NM.keys() if (
                    (req_info['arrival_interval'].loc[m] <= i) & (req_info['leave_interval'].loc[m] >= i))]
            # for _, req_li in X_MN.items():
            #     if len(req_li)!=0:
            #         previous_assigned.append(req_li[np.array([req_info['leave_interval'].loc[m] for m in req_li]).argmax()])
            # potential_conflict_assigned = min([req_info['arrival_interval'].loc[m] for m in previous_assigned])
            t_total_index = total_index + previous_assigned + adj_total_index

            plp = Model("linearized model")

            x_nm = plp.addVars({(n, m) for n in range(N) for m in t_total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
            y_mz = plp.addVars({(m, z) for m in total_index + adj_total_index for z in range(Z)}, vtype=GRB.BINARY,
                               name="y_mz")  # m*z
            # z_mm = plp.addVars({(m, m_) for m in t_total_index for m_ in t_total_index if m != m_}, vtype=GRB.BINARY,
            #                    name="z_mm")

            for m in t_total_index:
                for m_ in t_total_index:
                    if m != m_:
                        Z_MM[(m, m_)] = z_ik(arr[m_], lea[m], buffer)

            plp.update()

            # 平台收益：预约费 停车费 充电费
            park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
            char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))
            obj1 = park_revenue + char_revenue

            # 拒绝数量
            refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)
            refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)
            obj2 = refuse_park + refuse_char

            park_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in park_index + adj_park_index for z in range(Z))

            charge_travel = quicksum(
                cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] *
                y_mz[(m, z)] for m in charge_index + adj_charge_index for z in range(Z))

            obj3 = park_travel + charge_travel

            # 总目标
            # obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
            # model.setObjective(obj, GRB.MAXIMIZE)
            plp.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0,
                              priority=5)
            plp.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1,
                              priority=3)

            plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) == 1 for m in adj_total_index)

            plp.addConstrs(x_nm[(X_NM[assigned_driver], assigned_driver)] == 1 for assigned_driver in previous_assigned)

            plp.addConstrs(
                quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
                total_index + adj_total_index)

            plp.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

            # plp.addConstrs(
            #     x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + z_mm[(m, m_)] * Z_MM[(m, m_)] + z_mm[(m_, m)] * Z_MM[(m_, m)] for n
            #     in range(N) for m in t_total_index for m_ in t_total_index if m != m_)

            plp.addConstrs(
                x_nm[(n, m)] + x_nm[(n, m_)] <= 1 + Z_MM[(m, m_)] + Z_MM[(m_, m)] for n
                in range(N) for m in t_total_index for m_ in t_total_index if m != m_)

            plp.addConstrs(
                quicksum(x_nm[(n, m)] * r_mk[m][k] for m in t_total_index) <= 1 for n in range(N) for k in range(K))

            if rule == 0:
                # rule0
                # 停车请求只能分配到OPS
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index + adj_park_index)
            elif rule == 1:
                pass
            elif rule == 2:
                filtered_df = req_info.loc[park_index + adj_park_index]
                no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)

            # 6-10 14:18
            elif rule == 3:
                filtered_df = req_info.loc[park_index + adj_park_index]
                no_allocatable_index_1 = filtered_df[
                    (filtered_df['arrival_t'] < 360) | (filtered_df['arrival_t'] > 1020)].index.tolist()
                no_allocatable_index_2 = filtered_df[
                    (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
                plp.addConstrs(
                    quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in
                    no_allocatable_index_1 + no_allocatable_index_2)
            # 短时停车放快充 长时停车放慢充
            else:
                filtered_df = req_info.loc[park_index + adj_park_index]
                no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_1)
                no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
                plp.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_2)

            # 快充电请求只能分配到快充CPS
            plp.addConstrs(
                quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index + adj_fast_charge_index)
            plp.addConstrs(
                quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index + adj_fast_charge_index)

            # 慢充电请求只能分配到慢充CPS
            plp.addConstrs(
                quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index + adj_slow_charge_index)
            plp.addConstrs(
                quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index + adj_slow_charge_index)

            plp.optimize()

            res[1] += obj1.getValue()  # 停车场收益
            res[2] += park_revenue.getValue()  # 停车收益
            res[3] += char_revenue.getValue()  # 充电收益
            res[4] += obj2.getValue()  # 拒绝总数量
            res[5] += refuse_park.getValue()  # 停车拒绝数
            res[6] += refuse_char.getValue()  # 充电拒绝数
            res[7] += obj3.getValue()  # 行程时间

            assign_index = []

            for m in total_index:
                for n in range(N):
                    if x_nm[(n, m)].X == 1:
                        S_NK[n] -= r_mk[m]
                        S_NK_A[n] -= r_mk_a[m]
                        X_NM[m] = n
                        assign_index.append(m)
                        for z in range(Z):
                            if y_mz[(m, z)].X == 1:
                                Y_ZM[m] = z
                                assign_info.append([m, n, z, i])
                        break

            for a in adj_total_index:
                for n in range(N):
                    if x_nm[(n, a)].X == 1:
                        S_NK[n] -= r_mk[a]
                        S_NK_A[n] -= r_mk_a[a]
                        X_NM[a] = n
                        assign_index.append(a)
                        for z in range(Z):
                            if y_mz[(a, z)].X == 1:
                                Y_ZM[a] = z
                                assign_info.append([a, n, z, i])
                        break

            # 存储分配的信息
            # 将上一轮可以调整的请求和本轮可以调整的请求合并 继续调整
            # 本轮可以调整的请求索引
            this_adj_total_index = list(
                set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            this_adj_park_index = list(set(this_adj_total_index) & set(park_index))
            this_adj_charge_index = list(set(this_adj_total_index) & set(charge_index))
            this_adj_fast_charge_index = list(set(this_adj_charge_index) & set(fast_charge_index))
            this_adj_slow_charge_index = list(set(this_adj_charge_index) & set(slow_charge_index))

            # 上一轮可以继续调整的请求索引
            req_info.loc[adj_total_index, 'diff'] = req_info.loc[adj_total_index, 'diff'] - 1
            secondary_adj_total_index = [idx for idx in adj_total_index if req_info.at[idx, 'diff'] > 1]
            secondary_adj_park_index = list(set(secondary_adj_total_index) & set(adj_park_index))
            secondary_adj_charge_index = list(set(secondary_adj_total_index) & set(adj_charge_index))
            secondary_adj_fast_charge_index = list(set(secondary_adj_charge_index) & set(adj_fast_charge_index))
            secondary_adj_slow_charge_index = list(set(secondary_adj_charge_index) & set(adj_slow_charge_index))

            # 合并后作为总的可以调整的索引
            adj_total_index = secondary_adj_total_index + this_adj_total_index
            adj_park_index = secondary_adj_park_index + this_adj_park_index
            adj_charge_index = secondary_adj_charge_index + this_adj_charge_index
            adj_fast_charge_index = secondary_adj_fast_charge_index + this_adj_fast_charge_index
            adj_slow_charge_index = secondary_adj_slow_charge_index + this_adj_slow_charge_index

            # 恢复SNK
            for adj_index in adj_total_index:
                space_n = X_NM[adj_index]
                S_NK[space_n] += r_mk[adj_index]
                S_NK_A[space_n] += r_mk_a[adj_index]
                del X_NM[adj_index]

    # 计算收益
    # 对多次分配的请求删除合并，计算收益
    dpr_assign = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                              data=np.array(assign_info).reshape((-1, 4)))
    dpr_assign_filter = dpr_assign.loc[dpr_assign.groupby('req_id')['assign_t'].idxmax()]
    dpr_assign_filter.reset_index(drop=True, inplace=True)
    travel_cost = []
    for i in range(len(dpr_assign_filter)):
        temp_o = req_info['O'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_d = req_info['D'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_pl = dpr_assign_filter['pl_num'].iloc[i]
        travel_cost.append(cost_matrix_[int(temp_pl)][int(temp_o)] + 2 * cost_matrix_[int(temp_pl)][int(temp_d) + 2])
    dpr_assign_filter['travel_cost'] = travel_cost
    average_travel_cost = dpr_assign_filter['travel_cost'].sum() / len(dpr_assign_filter)
    res[-1] = average_travel_cost

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :96]) / (sum(pl[l].ordinary_num for l in range(Z)) * 96)
    charging_util = np.sum(S_NK[CPS_INDEX, :96]) / (sum(pl[l].charge_num for l in range(Z)) * 96)
    total_util = np.sum(S_NK[:, :96]) / (sum(pl[l].total_num for l in range(Z)) * 96)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    acc_num = req_num - res[4]
    P_acc = acc_num / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (charge_num - res[6]) / charge_num

    # 成功率
    faile_num = len(np.where(S_NK_A < 0)[0])
    success_park = (acc_num - faile_num) / acc_num

    # 冲突惩罚
    confilct_cost = 5 * faile_num
    # 净收益
    net_revenue = res[1] - confilct_cost

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "total_util": total_util, "travel cost": res[-1],
                   "success ratio": success_park,
                   "conflict cost": confilct_cost, 'net revenue': net_revenue}

    print("-------unp-rdp------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    # fig, axes = plt.subplots(1, 2)
    # vmin_ = min(S_NK.min(), S_NK_A.min())
    # sns.heatmap(S_NK[:, 31:47], ax=axes[0], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # sns.heatmap(S_NK_A[:, 31:47], ax=axes[1], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # axes[0].set_title('submitted request', fontsize=14)
    # axes[1].set_title('actual request', fontsize=14)
    # axes[1].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[0].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[1].set_xticks([i for i in range(0, 18, 2)], )
    # axes[0].set_xticks([i for i in range(0, 18, 2)])
    # axes[1].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # axes[0].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # # axes[0].set_xlabel('Decision interval', fontsize=14)
    # # axes[1].set_xlabel('Decision interval', fontsize=14)
    # axes[0].set_ylabel('parking space', fontsize=14)
    # # axes[1].set_ylabel('Parking space', fontsize=16)
    # plt.tight_layout()
    # # plt.savefig('fbfs_test.png', dpi=300)
    # plt.show()

    if need_save:

        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\unpanalysis')

        try:
            os.makedirs(f'{park_arrival_num}-{charge_num}-{version}')
        except:
            pass
        os.chdir(f'{park_arrival_num}-{charge_num}-{version}')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'SNK_info', 'SNKA_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/unp-rdp_{rule}_{buffer}.npy", S_NK)
        np.save(f"SNKA_info/aunp-rdp_{rule}_{buffer}.npy", S_NK_A)
        np.save(f"result_info/unp-rdp_{rule}_{buffer}.npy", result_dict)
        merge_data = pd.merge(dpr_assign_filter, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/unp-rdp_{rule}_{buffer}.csv", index=False)

        # fig, axes = plt.subplots(1, 2)
        # vmin_ = min(S_NK.min(), S_NK_A.min())
        # sns.heatmap(S_NK[:, :96], ax=axes[0], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
        #             vmin=vmin_, vmax=1, cbar=False)
        # sns.heatmap(S_NK_A[:, :96], ax=axes[1], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
        #             vmin=vmin_, vmax=1, cbar=False)
        # axes[0].set_title('submitted request', fontsize=16)
        # axes[1].set_title('actual request', fontsize=16)
        # # axes[1].set_yticklabels([i + 1 for i in range(100)], rotation=0, fontsize=12)
        # # axes[0].set_yticklabels([i + 1 for i in range(96)], rotation=0, fontsize=12)
        # axes[1].set_xticks([i for i in range(0, 96, 4)])
        # axes[0].set_xticks([i for i in range(0, 96, 4)])
        # # axes[1].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
        # # axes[0].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
        # # axes[0].set_xlabel('Decision interval', fontsize=16)
        # # axes[1].set_xlabel('Decision interval', fontsize=16)
        # # axes[0].set_ylabel('Parking space', fontsize=16)
        # # axes[1].set_ylabel('Parking space', fontsize=16)
        # plt.tight_layout()
        # # plt.savefig('rdp_unp.png', dpi=300)
        # plt.show()


def so(rule):
    """
    alpha1:收益系数
    alpha2:拒绝系数
    alpha3:时间系数
    """

    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []
    # 记录分配结果
    res = [0, 0, 0, 0, 0, 0, 0, 0]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；
    X_NM = defaultdict(int)
    Y_ZM = defaultdict(int)

    model = Model("so")

    total_index = req_info.index.tolist()
    park_index = req_info[req_info['label'] == 0].index.tolist()
    charge_index = req_info[req_info['label'] == 1].index.tolist()
    fast_charge_index = req_info[req_info['new_label'] == 1].index.tolist()
    slow_charge_index = req_info[req_info['new_label'] == 2].index.tolist()

    x_nm = model.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
    y_mz = model.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

    model.update()

    # 平台收益：预约费 停车费 充电费
    park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))
    char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))
    obj1 = park_revenue + char_revenue

    # 拒绝数量
    refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)
    refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)
    obj2 = refuse_park + refuse_char

    park_travel = quicksum(
        cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
            (m, z)] for m in park_index for z in range(Z))

    charge_travel = quicksum(
        cost_matrix_[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix_[z][req_info['D'].loc[m] + 2] * y_mz[
            (m, z)] for m in charge_index for z in range(Z))

    obj3 = park_travel + charge_travel

    # 总目标
    # obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
    # model.setObjective(obj, GRB.MAXIMIZE)
    model.setObjectiveN(-alpha1 * char_revenue + alpha2 * refuse_char + alpha3 * charge_travel, index=0, priority=5)
    model.setObjectiveN(-alpha1 * park_revenue + alpha2 * refuse_park + alpha3 * park_travel, index=1, priority=3)

    model.addConstrs(quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= S_NK[n][k] for n in
                     range(N) for k in range(K))

    model.addConstrs(
        quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_mz[(m, z)] for z in range(Z) for m in
        total_index)
    model.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

    if rule == 0:
        # rule0
        # 停车请求只能分配到OPS
        model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
    elif rule == 1:
        pass
    elif rule == 2:
        filtered_df = req_info.loc[park_index]
        no_allocatable_index = filtered_df[filtered_df['activity_t'] >= 450].index.tolist()
        model.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
    # 6-10 14:18
    elif rule == 3:
        filtered_df = req_info.loc[park_index]
        no_allocatable_index_1 = filtered_df[
            (filtered_df['arrival_t'] < 360) | (filtered_df['arrival_t'] > 1020)].index.tolist()
        no_allocatable_index_2 = filtered_df[
            (filtered_df['arrival_t'] > 540) & (filtered_df['arrival_t'] < 840)].index.tolist()
        model.addConstrs(
            quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index_1 + no_allocatable_index_2)
    # 短时停车放快充 长时停车放慢充
    else:
        filtered_df = req_info.loc[park_index]
        no_allocatable_index_1 = filtered_df[filtered_df['activity_t'] <= 90].index.tolist()
        model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in no_allocatable_index_1)
        no_allocatable_index_2 = filtered_df[filtered_df['activity_t'] > 90].index.tolist()
        model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in no_allocatable_index_2)

    # 快充电请求只能分配到快充CPS
    model.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
    model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

    # 慢充电请求只能分配到慢充CPS
    model.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
    model.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

    model.optimize()

    res[1] += obj1.getValue()  # 停车场收益
    res[2] += park_revenue.getValue()  # 停车收益
    res[3] += char_revenue.getValue()  # 充电收益
    res[4] += obj2.getValue()  # 拒绝总数量
    res[5] += refuse_park.getValue()  # 停车拒绝数
    res[6] += refuse_char.getValue()  # 充电拒绝数
    res[7] += obj3.getValue()  # 行程时间

    assign_index = []

    for m in total_index:
        for n in range(N):
            if x_nm[(n, m)].X == 1:
                S_NK[n] -= r_mk[m]
                S_NK_A[n] -= r_mk_a[m]
                X_NM[(n, m)] = 1
                assign_index.append(m)
                for z in range(Z):
                    if y_mz[(m, z)].X == 1:
                        Y_ZM[(z, m)] = 1
                        assign_info.append([m, n, z, i])
                break

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :96]) / (sum(pl[l].ordinary_num for l in range(Z)) * 96)
    charging_util = np.sum(S_NK[CPS_INDEX, :96]) / (sum(pl[l].charge_num for l in range(Z)) * 96)
    total_util = np.sum(S_NK[:, :96]) / (sum(pl[l].total_num for l in range(Z)) * 96)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    acc_num = req_num - res[4]
    P_acc = acc_num / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (charge_num - res[6]) / charge_num
    # travel cost
    res[7] /= acc_num

    # 成功率
    faile_num = len(np.where(S_NK_A < 0)[0])
    success_park = (acc_num - faile_num) / acc_num

    # 冲突惩罚
    confilct_cost = 5 * faile_num
    # 净收益
    net_revenue = res[1] - confilct_cost

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "total_util": total_util, "travel cost": res[-1],
                   "success ratio": success_park,
                   "conflict cost": confilct_cost, 'net revenue': net_revenue}

    print("-------so-------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    # fig, axes = plt.subplots(1, 2)
    # sns.heatmap(S_NK[:, :96], ax=axes[0])
    # sns.heatmap(S_NK_A[:, :96], ax=axes[1])
    # axes[0].set_title('Submitted')
    # axes[1].set_title('Actual')
    # # plt.title(f"unp-dp:{rule}")
    # plt.show()

    if need_save:

        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\save_data_1121')

        try:
            os.makedirs(f'{park_arrival_num}-1')
        except:
            pass
        os.chdir(f'{park_arrival_num}-1')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'SNK_info', 'SNKA_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/so_{rule}.npy", S_NK)
        np.save(f"SNKA_info/aso_{rule}.npy", S_NK_A)
        np.save(f"result_info/so_{rule}.npy", result_dict)
        assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                        data=np.array(assign_info).reshape((-1, 4)))
        merge_data = pd.merge(assign_info_data, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/so_{rule}.csv", index=False)


def fbfs(rule):
    """
    alpha1:收益系数
    alpha2:拒绝系数
    alpha3:时间系数
    """
    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []
    # 记录分配结果
    res = [0, 0, 0, 0, 0, 0, 0, 0]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；
    X_NM = defaultdict(int)
    Y_ZM = defaultdict(int)

    req_info.sort_values(by='request_t', inplace=True)
    req_info_index = req_info.index.tolist()

    for i in req_info_index:
        print("-------------------第" + str(i) + "次分配结果-------------------")

        model = Model("fbfs")

        # 根据请求类型分配到普通泊位/充电桩
        label = req_info['new_label'].loc[i]
        arrival_t = req_info['arrival_t'].loc[i]
        activity_t = req_info['activity_t'].loc[i]

        x_nm = model.addVars({(n, 0) for n in range(N)}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_mz = model.addVars({(0, z) for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        model.update()

        # 平台收益：预约费 停车费 充电费
        obj1 = quicksum(x_nm[(n, 0)] * req_info['revenue'].loc[i] for n in range(N))

        # 拒绝数量
        obj2 = 1 - quicksum(x_nm[(n, 0)] for n in range(N))

        obj3 = quicksum(
            cost_matrix_[z][req_info['O'].loc[i]] * y_mz[(0, z)] + 2 * cost_matrix_[z][req_info['D'].loc[i] + 2] * y_mz[
                (0, z)] for z in range(Z))

        # obj3 = cost_matrix_[0][req_info['O'].loc[i]] * y_mz[(0, 0)] + 2 * cost_matrix_[0][req_info['D'].loc[i] + 2] * \
        #        y_mz[(0, 0)]

        # 总目标
        obj = obj3 - 100 * (1 - obj2)
        model.setObjective(obj, GRB.MINIMIZE)

        model.addConstrs(x_nm[(n, 0)] * r_mk[i][k] <= S_NK[n][k] for n in range(N) for k in range(K))

        model.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n, 0)] for n in range(N)) == y_mz[(0, z)] for z in range(Z))

        model.addConstr(quicksum(x_nm[(n, 0)] for n in range(N)) <= 1)

        if label == 0:
            if rule == 0:
                # rule0
                # 停车请求只能分配到OPS
                model.addConstr(quicksum(x_nm[(n, 0)] for n in CPS_INDEX) == 0)
            elif rule == 1:
                pass
            elif rule == 2:
                if activity_t >= 450:
                    model.addConstr(quicksum(x_nm[(n, 0)] for n in CPS_INDEX) == 0)
            # 6-10 14:18
            elif rule == 3:
                if (arrival_t < 360) | (540 < arrival_t < 840) | (arrival_t > 1020):
                    model.addConstr(quicksum(x_nm[(n, 0)] for n in CPS_INDEX) == 0)
            # 短时停车放快充 长时停车放慢充
            else:
                if activity_t <= 90:
                    model.addConstr(quicksum(x_nm[(n, 0)] for n in FAST_INDEX) == 0)
                else:
                    model.addConstr(quicksum(x_nm[(n, 0)] for n in SLOW_INDEX) == 0)

        else:
            if label == 1:
                # 快充电请求只能分配到快充CPS
                model.addConstr(quicksum(x_nm[(n, 0)] for n in SLOW_INDEX) == 0)
                model.addConstr(quicksum(x_nm[(n, 0)] for n in OPS_INDEX) == 0)
            else:
                # 慢充电请求只能分配到慢充CPS
                model.addConstr(quicksum(x_nm[(n, 0)] for n in FAST_INDEX) == 0)
                model.addConstr(quicksum(x_nm[(n, 0)] for n in OPS_INDEX) == 0)

        model.optimize()

        res[0] += obj.getValue()  # 目标函数
        res[1] += obj1.getValue()  # 停车场收益
        res[4] += obj2.getValue()  # 拒绝总数量
        if label == 0:
            res[2] += obj1.getValue()  # 停车收益
            res[5] += obj2.getValue()  # 停车拒绝数
        else:
            res[3] += obj1.getValue()  # 充电收益
            res[6] += obj2.getValue()  # 充电拒绝数
        res[7] += obj3.getValue()  # 行程时间

        assign_index = []

        for n in range(N):
            if x_nm[(n, 0)].X == 1:
                S_NK[n] -= r_mk[i]
                S_NK_A[n] -= r_mk_a[i]
                X_NM[(n, i)] = 1
                assign_index.append(i)
                for z in range(Z):
                    if y_mz[(0, z)].X == 1:
                        Y_ZM[(z, i)] = 1
                        assign_info.append([i, n, z, req_info['request_interval'].loc[i]])

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :96]) / (sum(pl[l].ordinary_num for l in range(Z)) * 96)
    charging_util = np.sum(S_NK[CPS_INDEX, :96]) / (sum(pl[l].charge_num for l in range(Z)) * 96)
    total_util = np.sum(S_NK[:, :96]) / (sum(pl[l].total_num for l in range(Z)) * 96)

    # acceptance rate of reservation requests
    req_num = len(req_info)
    acc_num = req_num - res[4]
    P_acc = acc_num / req_num
    P_park_acc = (park_arrival_num - res[5]) / park_arrival_num
    P_char_acc = (charge_num - res[6]) / charge_num
    # travel cost
    res[7] /= acc_num

    # 成功率
    faile_num = len(np.where(S_NK_A < 0)[0])
    success_park = (acc_num - faile_num) / acc_num

    # 冲突惩罚
    confilct_cost = 5 * faile_num
    # 净收益
    net_revenue = res[1] - confilct_cost

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "request number": req_num,
                   "objective value": res[0], "total revenue": res[1],
                   "park revenue": res[2], "char revenue": res[3], "refuse number": res[4],
                   "refuse park number": res[5], "refuse char number": res[6],
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, 'total_util': total_util, "travel cost": res[-1],
                   "success ratio": success_park,
                   "conflict cost": confilct_cost, 'net revenue': net_revenue}

    print("------fbfs------")
    for key, value in result_dict.items():
        print(key + ": " + str(value))

    # fig, axes = plt.subplots(1, 2)
    # vmin_ = min(S_NK.min(), S_NK_A.min())
    # sns.heatmap(S_NK[:, 31:47], ax=axes[0], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # sns.heatmap(S_NK_A[:, 31:47], ax=axes[1], cmap="Accent", linewidths=0.05, linecolor='white', square=True,
    #             vmin=vmin_, vmax=1, cbar=False)
    # axes[0].set_title('submitted request', fontsize=14)
    # axes[1].set_title('actual request', fontsize=14)
    # axes[1].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[0].set_yticklabels([i + 1 for i in range(10)], rotation=0, fontsize=12)
    # axes[1].set_xticks([i for i in range(0, 18, 2)], )
    # axes[0].set_xticks([i for i in range(0, 18, 2)])
    # axes[1].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # axes[0].set_xticklabels([i for i in range(32, 50, 2)], fontsize=12)
    # # axes[0].set_xlabel('Decision interval', fontsize=14)
    # # axes[1].set_xlabel('Decision interval', fontsize=14)
    # axes[0].set_ylabel('parking space', fontsize=14)
    # # axes[1].set_ylabel('Parking space', fontsize=16)
    # plt.tight_layout()
    # # plt.savefig('fbfs_test.png', dpi=300)
    # plt.show()

    if need_save:
        # 保存数据
        os.chdir(r'G:\2023-纵向\停车分配\save_data_1121')

        try:
            os.makedirs(f'{park_arrival_num}-1')
        except:
            pass
        os.chdir(f'{park_arrival_num}-1')

        # 创建子文件夹
        folder_name = ['assign_info', 'result_info', 'SNK_info', 'SNKA_info']
        for each_folder in folder_name:
            try:
                os.makedirs(each_folder)
            except:
                pass

        np.save(f"SNK_info/fbfs_{rule}.npy", S_NK)
        np.save(f"SNKA_info/afbfs_{rule}.npy", S_NK_A)
        np.save(f"result_info/fbfs_{rule}.npy", result_dict)
        assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                        data=np.array(assign_info).reshape((-1, 4)))
        merge_data = pd.merge(assign_info_data, req_info, on='req_id', how='left')
        merge_data.to_csv(f"assign_info/fbfs_{rule}.csv", index=False)


def save_SNK_A(assign_info, rmka, SNKA):
    for i in range(len(assign_info)):
        req_m = assign_info['req_id'].iloc[i]
        space_n = assign_info['space_num'].iloc[i]
        SNKA[space_n] -= rmka[req_m]
    return SNKA


if __name__ == '__main__':

    # OD及停车场信息
    O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量
    cost_matrix_ = OD.OdCost().cost_matrix
    # pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z, config='1:1')
    # pl = [pl1, pl2, pl3, pl4]
    #
    park_fee = 6 / 2  # 半个小时的费用
    charge_fee = [0, 0.7, 0.5]  # 每分钟的价格
    reserved_fee = 1  # 预约费用
    #
    N = 100  # 总泊位数
    # N = 10
    # Z = 2

    # 泊位索引
    # T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
    # _, OPS_INDEX = P_ZN(Z, N, pl)
    # _, CPS_INDEX = C_ZN(Z, N, pl)
    # _, FAST_INDEX = Fast_ZN(Z, N, pl)
    # _, SLOW_INDEX = Slow_ZN(Z, N, pl)

    # T_ZN = np.array(
    #     ([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])).reshape(2, -1)
    # OPS_INDEX = [0, 1, 2, 3, 4]
    # CPS_INDEX = [5, 6, 7, 8, 9]
    # FAST_INDEX = [5,6,7,8]
    # SLOW_INDEX = [9]

    # 绘图设置
    # mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['font.serif'] = ['Times New Roman']
    #
    # """
    # sensitivity analysis
    # """
    ratio_31 = [i for i in range(30, 271, 30)]
    ratio_61 = [i for i in range(60, 541, 60)]

    total_ratio = [ratio_31, ratio_61]

    for conf in [i / 10 for i in range(2, 10)]:
        pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z, config=conf)
        pl = [pl1, pl2, pl3, pl4]
        # 泊位索引
        T_ZN, _, EACH_INDEX = T_ZN_(Z, N, pl)
        _, OPS_INDEX = P_ZN(Z, N, pl)
        _, CPS_INDEX = C_ZN(Z, N, pl)
        _, FAST_INDEX = Fast_ZN(Z, N, pl)
        _, SLOW_INDEX = Slow_ZN(Z, N, pl)
        for j in [0, 1, 2]:
            for i in range(0, 2):
                ratio_ = total_ratio[i]
                for park_num in ratio_:
                    if i == 0:
                        req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand1125\\{park_num}-{300 - park_num}.csv")
                        charge_num = 300 - park_num
                    else:
                        req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand1125\\{park_num}-{600 - park_num}.csv")
                        charge_num = 600 - park_num

                    park_arrival_num = park_num

                    decision_interval = 15
                    req_info = update_req_info(req_info)

                    req_info['request_interval'] = req_info['request_t'] // decision_interval
                    req_info['arrival_interval'] = req_info['arrival_t'] // decision_interval
                    req_info['leave_interval'] = req_info['leave_t'] // decision_interval

                    K = max(req_info['leave_interval'])  # 时间长度
                    r_mk = get_rmk(req_info).astype(int)
                    r_mk_a = get_rmk_a(req_info).astype(int)

                    request_interval = req_info['request_interval'].unique()

                    # 计算达到到的时间间隔
                    req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']

                    # 迭代次数
                    I = request_interval
                    I.sort()

                    need_save = True
                    alpha1 = 1
                    alpha2 = 1
                    alpha3 = 1

                    S_NK = np.ones((N, K)).astype(int)
                    S_NK_A = np.ones((N, K)).astype(int)

                    rdp(rule=j, ops_num=conf)

            # for sl in range(15, 151, 15):
            #     S_NK = np.ones((N, K)).astype(int)
            #     S_NK_A = np.ones((N, K)).astype(int)
            #     rdp(rule=2, parking_duration=pkd)
            #     rdp(rule=4, sl_cut=sl)
            #     req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']

    # for j in range(4, 5):
    # for b in range(0,31,5):
    #     for i in range(150, 301, 150):
    #         for v in range(2):
    #             # 需求信息
    #             park_arrival_num = i
    #             charge_num = i
    #             req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand1125\\{park_arrival_num}-{charge_num}-{v}.csv")
    #             # req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\{park_arrival_num}-1.csv")
    #
    #             # 更新实际时间
    #             decision_interval = 15
    #             req_info = update_req_info(req_info)
    #
    #             req_info['request_interval'] = req_info['request_t'] // decision_interval
    #             req_info['arrival_interval'] = req_info['arrival_t'] // decision_interval
    #             req_info['leave_interval'] = req_info['leave_t'] // decision_interval
    #
    #             K = max(req_info['leave_interval'])  # 时间长度
    #             r_mk = get_rmk(req_info).astype(int)
    #             r_mk_a = get_rmk_a(req_info).astype(int)
    #
    #             # req_info = req_info.loc[(req_info['request_interval'] >= 32) & (req_info['request_interval'] <= 48)]
    #             # park_info = req_info.loc[req_info['new_label'] == 0].sample(n=25, random_state=43)
    #             # charge_info = req_info.loc[req_info['new_label'] == 1].sample(n=25, random_state=43)
    #             # req_info = pd.concat([park_info, charge_info], axis=0)
    #             request_interval = req_info['request_interval'].unique()
    #
    #             # 计算达到到的时间间隔
    #             req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']
    #
    #             # 迭代次数
    #             I = request_interval
    #             I.sort()
    #
    #             need_save = True
    #             alpha1 = 1
    #             alpha2 = 1
    #             alpha3 = 1
    #
    #             S_NK = np.ones((N, K)).astype(int)
    #             S_NK_A = np.ones((N, K)).astype(int)
    #
    #             # park_arrival_num = 25
    #             # charge_num = 25
    #
    #             # rdp(rule=j)
    #             # fbfs(rule=j)
    #             # dp(rule=j)
    #             # rdp_unpunctuality(rule=4,buffer=b,version=v)

    # for b in [0.33, 0.5, 1, 1.33, 1.67, 2]:
    # for b in [0, 5, 10, 15, 20, 25, 30]:
    #     S_NK = np.ones((N, K)).astype(int)
    #     S_NK_A = np.ones((N, K)).astype(int)
    #     rdp_unpunctuality(rule=j, given_buffer=b)
    #     req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']
