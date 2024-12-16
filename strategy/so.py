"""
系统最优
将所有信息一次性分配
"""
import pandas as pd
from entity import OD
from utils import *
from gurobipy import *


def so(rule, alpha1=1, alpha2=1, alpha4=5):
    r_mk = get_rmk(req_info).astype(int)
    assign_info = []

    m = Model("so")
    m.Params.OutputFlag = 1

    total_index = req_info.index.tolist()
    park_index = req_info[req_info['label'] == 0].index.tolist()
    charge_index = req_info[req_info['label'] == 1].index.tolist()
    fast_charge_index = req_info[req_info['new_label'] == 1].index.tolist()
    slow_charge_index = req_info[req_info['new_label'] == 2].index.tolist()

    x_nm = m.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
    y_zm = m.addVars({(z, m) for z in range(Z) for m in total_index}, vtype=GRB.BINARY, name="y_zm")  # z*m
    y_mz = m.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

    m.update()

    # 平台收益： 预约费用 + 停车费用 + 充电费用
    park_revenue = quicksum(x_nm[(n, m)] * req_info['std_revenue'].loc[m] for m in park_index for n in range(N))
    park_revenue_ = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))

    char_revenue = quicksum(x_nm[(n, m)] * req_info['std_revenue'].loc[m] for m in charge_index for n in range(N))
    char_revenue_ = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))

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
    obj = alpha1 * obj1 - alpha2 * obj2 - alpha4 * obj4
    # obj = alpha2 * obj2

    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstrs(y_zm[(z, m)] == y_mz[(m, z)] for z in range(Z) for m in total_index)

    m.addConstrs(
        quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= 1 for n in range(N) for k in range(K))

    m.addConstrs(
        quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_zm[(z, m)] for z in range(Z) for m in total_index)
    m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

    if rule == 1:
        # rule1
        # 停车请求只能分配到OPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in park_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
    elif rule == 2:
        # rule2:
        # 停车请求可以分配到OPS和CPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in park_index)
        pass
    elif rule == 3:
        # 只有在规定时间内停车请求可以分配到充电桩
        # 8-10 17-19 可以停放（到达时间为480-600或1020-1140）
        # 初步筛选出 park_index 对应的行
        filtered_df = req_info.loc[park_index]
        # 在初步筛选出的行中，进一步筛选出 arrival_t 在 480 和 600 之间的行，并获取这些行的原始索引
        allocatable_index_1 = filtered_df[
            (filtered_df['arrival_t'] >= 480) & (filtered_df['arrival_t'] <= 600)].index.tolist()
        # 在初步筛选出的行中，进一步筛选出 arrival_t 在 1020 和 1140 之间的行，并获取这些行的原始索引
        allocatable_index_2 = filtered_df[
            (filtered_df['arrival_t'] >= 1020) & (filtered_df['arrival_t'] <= 1140)].index.tolist()
        # 合并两个列表
        allocatable_index = allocatable_index_1 + allocatable_index_2
        # 符合条件的停车请求可以分配到OPS和CPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in allocatable_index)
        # 不符合的只能分配到OPS
        no_allocatable_index = list(set(park_index) - set(allocatable_index))
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in no_allocatable_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
    elif rule == 4:
        # 短时停车停放在快充桩 长时停车停放在慢充桩
        # 短时停车(<=30分钟) 长时停车(30-150)
        # 筛选出 park_index 对应的行
        filtered_df = req_info.loc[park_index]
        # 在筛选出的行中，进一步筛选 activity_t <= 30 的行，并获取这些行的原始索引
        allocatable_short_index = filtered_df[filtered_df['activity_t'] <= 30].index.tolist()
        # 在筛选出的行中，进一步筛选 30 < activity_t <= 150 的行，并获取这些行的原始索引
        allocatable_long_index = filtered_df[
            (filtered_df['activity_t'] > 30) & (filtered_df['activity_t'] <= 150)].index.tolist()
        no_allocatable_index = list(set(park_index) - set(allocatable_short_index) - set(allocatable_long_index))

        try:
            # 短时停车请求分配到快充电桩
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in (OPS_INDEX + FAST_INDEX)) <= 1 for m in allocatable_short_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in allocatable_short_index)
        except:
            pass
        try:
            # 长时停车请求分配到慢充桩
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in (OPS_INDEX + SLOW_INDEX)) <= 1 for m in allocatable_long_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in allocatable_long_index)
        except:
            pass
        # 其他请求分配到OPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in no_allocatable_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
    else:
        # rule3和rule4的结合
        # 初步筛选出 park_index 对应的行
        filtered_df = req_info.loc[park_index]
        # 在初步筛选出的行中，进一步筛选出 arrival_t 在 480 和 600 之间的行，并获取这些行的原始索引
        allocatable_index_1 = filtered_df[
            (filtered_df['arrival_t'] >= 480) & (filtered_df['arrival_t'] <= 600)].index.tolist()
        # 在初步筛选出的行中，进一步筛选出 arrival_t 在 1020 和 1140 之间的行，并获取这些行的原始索引
        allocatable_index_2 = filtered_df[
            (filtered_df['arrival_t'] >= 1020) & (filtered_df['arrival_t'] <= 1140)].index.tolist()
        allocatable_time_index = allocatable_index_1 + allocatable_index_2

        # 在筛选出的行中，进一步筛选 activity_t <= 30 的行，并获取这些行的原始索引
        allocatable_short_activity_index = filtered_df[filtered_df['activity_t'] <= 30].index.tolist()
        # 在筛选出的行中，进一步筛选 30 < activity_t <= 150 的行，并获取这些行的原始索引
        allocatable_long_activity_index = filtered_df[
            (filtered_df['activity_t'] > 30) & (filtered_df['activity_t'] <= 150)].index.tolist()
        allocatable_short_index = list(set(allocatable_time_index) & set(allocatable_short_activity_index))
        allocatable_long_index = list(set(allocatable_time_index) & set(allocatable_long_activity_index))
        no_allocatable_index = list(set(park_index) - set(allocatable_short_index) - set(allocatable_long_index))
        try:
            # 短时停车请求分配到快充电桩
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in (OPS_INDEX + FAST_INDEX)) <= 1 for m in allocatable_short_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in allocatable_short_index)
        except:
            pass
        try:
            # 长时停车请求分配到慢充桩
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in (OPS_INDEX + SLOW_INDEX)) <= 1 for m in allocatable_long_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in allocatable_long_index)
        except:
            pass
        # 其他请求分配到OPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in no_allocatable_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)

    # 快充电请求只能分配到快充CPS
    # m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) <= 1 for m in fast_charge_index)
    m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
    m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
    # 慢充电请求只能分配到慢充CPS
    # m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) <= 1 for m in slow_charge_index)
    m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
    m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

    m.optimize()

    gc.collect()

    REVENUE_total[0] = obj.getValue()
    REVENUE_total[1] = obj1_.getValue()
    REVENUE_total[2] = park_revenue_.getValue()
    REVENUE_total[3] = char_revenue_.getValue()
    REVENUE_total[4] = obj4.getValue()
    REVENUE_total[5] = refuse_park.getValue()
    REVENUE_total[6] = refuse_char.getValue()
    REVENUE_total[7] = obj2_.getValue()


    X_NM = np.zeros((N, len(total_index)))
    for n in range(N):
        for temp_index, m_ in enumerate(total_index):
            X_NM[n][temp_index] = x_nm[(n, m_)].X
            if x_nm[(n, m_)].X == 1:
                for z in range(Z):
                    if y_zm[(z, m_)].X == 1:
                        assign_info.append([m_, n, z, req_info['request_interval'].loc[m_]])
                        break

    S_NK = np.matmul(X_NM, r_mk)

    obj4 = quicksum(
        pl_occ(z, S_NK, EACH_INDEX[z], req_info['arrival_t'].loc[m]) * y_mz[(m, z)] for m in total_index for z in
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
    # P_cruising = REVENUE_total[8]

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[OPS_INDEX, :1440]) / (sum(pl[i].ordinary_num for i in range(Z)) * 1440)
    charging_util = np.sum(S_NK[CPS_INDEX, :1440]) / (sum(pl[i].charge_num for i in range(Z)) * 1440)


    # acceptance rate of reservation requests
    req_num = len(r_mk)
    P_acc = (req_num - P_refuse) / req_num
    P_park_acc = (park_arrival_num - P_park_refuse) / park_arrival_num
    P_char_acc = (park_arrival_num * charge_ratio - P_char_refuse) / (park_arrival_num * charge_ratio)

    # 平均行程时间
    P_travel = P_travel / (req_num - P_refuse)
    # 平均巡游时间
    # P_cruising = P_cruising / (req_num - P_refuse)

    temp_occ = np.array([np.sum(S_NK[EACH_INDEX[z]], axis=0) / pl[z].total_num for z in range(Z)]).reshape(Z, K)
    P_occ_diff = np.mean(np.sum((temp_occ - np.mean(temp_occ, axis=0)) ** 2 / Z, axis=0))

    result_dict = {"alpha1": alpha1, "alpha2": alpha2, "alpha4": alpha4,"request number": req_num,
                   "objective value": P_obj, "total revenue": P_rev,
                   "park revenue": P_park_rev, "char revenue": P_char_rev, "refuse number": P_refuse,
                   "refuse park number": P_park_refuse, "refuse char number": P_char_refuse,
                   "acc": P_acc, "park acc": P_park_acc, "charge acc": P_char_acc, "park util": parking_util,
                   "charge util": charging_util, "travel cost": P_travel, "parking lot occ diff": P_occ_diff}

    for key, value in result_dict.items():
        print(key + ": " + str(value))


    # 保存数据
    os.chdir(r'G:\2023-纵向\停车分配\save_data_0923\需求11供给11')

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

    np.save(f"SNK_info/so_{rule}.npy", S_NK)
    np.save(f"revenue_info/so_{rule}.npy", REVENUE_total)
    np.save(f"result_info/so_{rule}.npy", result_dict)
    assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                    data=np.array(assign_info).reshape((-1, 4)))
    assign_info_data.to_csv(f"assign_info/so_{rule}.csv", index=False)


if __name__ == '__main__':
    global park_arrival_num
    charge_ratio = 1
    # OD及停车场信息
    O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
    cost_matrix_ = OD.OdCost().cost_matrix
    cost_matrix = OD.OdCost().get_std_cost()
    pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z, config='1:1')
    pl = [pl1, pl2, pl3, pl4]

    park_fee = pl1.park_fee / 2  # 半个小时的费用
    charge_fee = [0, pl1.fast_charge_fee, pl1.slow_charge_fee]  # 每分钟的价格
    reserved_fee = pl1.reserve_fee  # 预约费用

    N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数

    # 泊位索引
    T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
    _, OPS_INDEX = P_ZN(Z, N, pl)
    _, CPS_INDEX = C_ZN(Z, N, pl)
    _, FAST_INDEX = Fast_ZN(Z, N, pl)
    _, SLOW_INDEX = Slow_ZN(Z, N, pl)

    REVENUE_total = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0目标函数 1总收益 2停车收益 3充电收益 4拒绝总数 5停车拒绝数 6充电拒绝数 7出行时间 8巡游时间

    for i in range(50,525,25):
        park_arrival_num = i
        req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\{park_arrival_num}-{charge_ratio}.csv")

        # 将请求排序 并计算所在的间隔
        decision_interval = 15
        earliest_request = min(req_info['request_t'])
        req_info['request_interval'] = (req_info['request_t'] - earliest_request) // decision_interval
        request_interval = req_info['request_interval'].unique()

        req_info['revenue'] = [(np.floor(req_info['activity_t'].iloc[i] / 15) * park_fee + (
                req_info['activity_t'].iloc[i] * (charge_fee[req_info['new_label'].iloc[i]])) + reserved_fee) for i in
                               range(len(req_info))]
        req_info['std_revenue'] = (req_info['revenue'] - min(req_info['revenue'])) / (
                    max(req_info['revenue']) - min(req_info['revenue']))

        K = max(req_info['leave_t'])  # 时间长度

        so(rule=2)


