"""
先预约先服务的拓展
预约后在未到达之前可以调整 必须确保在下个决策间隔有泊位
决策间隔15min
"""
from datetime import datetime
from entity import OD
from utils import *
from gurobipy import *
import pandas as pd


def rdp(rule, alpha1=1, alpha2=1, alpha3=0):
    r_mk = get_rmk_(req_info).astype(int)  # 注意区别于get_rmk
    # 存储请求id 泊位 停车场 巡游时间 分配时所在的间隔 (req_id space_num pl_num assign_t)
    assign_info = []

    # 第一轮分配不用调整
    need_adjust = False
    X_NA_LAST_ = []
    # 存储上一轮已经分配完成且可以调整的请求索引
    adj_total_index = []
    adj_park_index = []
    adj_charge_index = []
    adj_fast_charge_index = []
    adj_slow_charge_index = []
    for ith, i in enumerate(request_interval):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        m = Model("rdp")
        m.Params.OutputFlag = 1

        # 本轮的请求
        this_req_info = req_info[req_info['request_interval'] == i]
        total_index = this_req_info.index.tolist()
        park_index = this_req_info[this_req_info['label'] == 0].index.tolist()
        charge_index = this_req_info[this_req_info['label'] == 1].index.tolist()
        fast_charge_index = this_req_info[this_req_info['new_label'] == 1].index.tolist()
        slow_charge_index = this_req_info[this_req_info['new_label'] == 2].index.tolist()

        X_NM = np.zeros((N, len(this_req_info))).astype(int)  # 该请求的分配情况
        R_MK = r_mk[total_index].reshape((len(this_req_info), K))  # 该请求
        # X_NA_LAST = X_NA_LAST_

        x_nm = m.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_zm = m.addVars({(z, m) for z in range(Z) for m in total_index}, vtype=GRB.BINARY, name="y_zm")  # z*m
        y_mz = m.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        if need_adjust:
            # 上轮可调整的请求在上一轮的分配结果
            # X_NA_LAST = X_NA_LAST_
            X_NA = np.zeros((N, len(adj_total_index))).astype(int)  # 该请求本轮的分配情况
            R_AK = r_mk[adj_total_index].reshape((len(adj_total_index), K))  # 该请求

            x_na = m.addVars({(n, a) for n in range(N) for a in adj_total_index}, vtype=GRB.BINARY, name="x_na")  # n*m
            y_za = m.addVars({(z, a) for z in range(Z) for a in adj_total_index}, vtype=GRB.BINARY, name="y_za")  # z*m
            y_az = m.addVars({(a, z) for a in adj_total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_az")  # m*z

        m.update()

        # 平台收益：预约费 停车费 充电费
        park_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in park_index for n in range(N))

        char_revenue = quicksum(x_nm[(n, m)] * req_info['revenue'].loc[m] for m in charge_index for n in range(N))

        obj1 = park_revenue + char_revenue

        # 拒绝数量
        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)

        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)

        obj2 = len(total_index) - quicksum(x_nm[(n, m)] for m in total_index for n in range(N))

        # 行程时间
        obj3 = quicksum(
            cost_matrix[z][req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix[z][req_info['D'].loc[m] + 2] * y_mz[
                (m, z)] for m in total_index for z in range(Z)) + \
               need_adjust * quicksum(
            cost_matrix[z][req_info['O'].loc[a]] * y_az[(a, z)] + 2 * cost_matrix[z][req_info['D'].loc[a] + 2] * y_az[
                (a, z)] for a in adj_total_index for z in range(Z))

        # 总目标
        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
        # obj = alpha2 * obj2

        m.setObjective(obj, GRB.MAXIMIZE)

        if need_adjust:
            if X_NA_LAST_.shape[1] != 0:
                m.addConstrs(y_za[(z, a)] == y_az[(a, z)] for z in range(Z) for a in adj_total_index)
                S_NK[ith] = S_NK[ith] - np.matmul(X_NA_LAST_, R_AK)

            m.addConstrs(
                S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in total_index) + quicksum(
                    x_na[(n, a)] * r_mk[a][k] for a in adj_total_index) <= 1 for n in range(N) for k in
                range(K))

            m.addConstrs(
                quicksum(T_ZN[z][n] * x_na[(n, a)] for n in range(N)) == y_za[(z, a)] for z in range(Z) for a in
                adj_total_index)

            m.addConstrs(quicksum(x_na[(n, a)] for n in range(N)) == 1 for a in adj_total_index)

        m.addConstrs(y_zm[(z, m)] == y_mz[(m, z)] for z in range(Z) for m in total_index)
        m.addConstrs(
            quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_zm[(z, m)] for z in range(Z) for m in
            total_index)
        m.addConstrs((1 - need_adjust) * S_NK[ith][n][k] + (1 - need_adjust) * quicksum(
            x_nm[(n, m)] * r_mk[m][k] for m in total_index) <= 1 for n in
                     range(N) for k in range(K))
        m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

        if rule == 1:
            # rule1
            # 停车请求只能分配到OPS
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in park_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
            # m.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 1 for a in adj_park_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in adj_park_index)

        elif rule == 2:
            # rule2:
            # 停车请求可以分配到OPS和CPS
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in park_index)

            # m.addConstrs(quicksum(x_na[(n, a)] for n in range(N)) == 1 for a in adj_park_index)
            pass
        elif rule == 3:
            # 只有在规定时间内停车请求可以分配到充电桩
            # 8-10 17-19 可以停放（到达时间为480-600或1020-1140）
            # 初步筛选出 park_index 对应的行
            filtered_df = req_info.loc[park_index]
            allocatable_index_1 = filtered_df[
                (filtered_df['arrival_t'] >= 480) & (filtered_df['arrival_t'] <= 600)].index.tolist()
            allocatable_index_2 = filtered_df[
                (filtered_df['arrival_t'] >= 1020) & (filtered_df['arrival_t'] <= 1140)].index.tolist()
            allocatable_index = allocatable_index_1 + allocatable_index_2
            no_allocatable_index = list(set(park_index) - set(allocatable_index))

            filtered_df_adj = req_info.loc[adj_park_index]
            allocatable_index_1_adj = filtered_df_adj[
                (filtered_df_adj['arrival_t'] >= 480) & (filtered_df_adj['arrival_t'] <= 600)].index.tolist()
            allocatable_index_2_adj = filtered_df_adj[
                (filtered_df_adj['arrival_t'] >= 1020) & (filtered_df_adj['arrival_t'] <= 1140)].index.tolist()
            allocatable_index_adj = allocatable_index_1_adj + allocatable_index_2_adj
            no_allocatable_index_adj = list(set(adj_park_index) - set(allocatable_index_adj))

            # 符合条件的停车请求可以分配到OPS和CPS
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in allocatable_index)
            # 不符合的只能分配到OPS
            # m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in no_allocatable_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in no_allocatable_index_adj)
        elif rule == 4:
            # 短时停车停放在快充桩 长时停车停放在慢充桩
            # 短时停车(<=30分钟) 长时停车(30-150)
            filtered_df = req_info.loc[park_index]
            allocatable_short_index = filtered_df[filtered_df['activity_t'] <= 30].index.tolist()
            allocatable_long_index = filtered_df[
                (filtered_df['activity_t'] > 30) & (filtered_df['activity_t'] <= 150)].index.tolist()
            no_allocatable_index = list(set(park_index) - set(allocatable_short_index) - set(allocatable_long_index))

            filtered_df_adj = req_info.loc[adj_park_index]
            allocatable_short_index_adj = filtered_df_adj[filtered_df_adj['activity_t'] <= 30].index.tolist()
            allocatable_long_index_adj = filtered_df_adj[
                (filtered_df_adj['activity_t'] > 30) & (filtered_df_adj['activity_t'] <= 150)].index.tolist()
            no_allocatable_index_adj = list(
                set(adj_park_index) - set(allocatable_short_index_adj) - set(allocatable_long_index_adj))

            try:
                # 短时停车请求分配到快充电桩
                m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in allocatable_short_index)
                m.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in allocatable_short_index_adj)
            except:
                pass
            try:
                # 长时停车请求分配到慢充桩
                m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in allocatable_long_index)
                m.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in allocatable_long_index_adj)
            except:
                pass
            # 其他请求分配到OPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in no_allocatable_index_adj)
        else:
            # rule3和rule4的结合
            # 初步筛选出 park_index 对应的行
            filtered_df = req_info.loc[park_index]
            allocatable_index_1 = filtered_df[
                (filtered_df['arrival_t'] >= 480) & (filtered_df['arrival_t'] <= 600)].index.tolist()
            allocatable_index_2 = filtered_df[
                (filtered_df['arrival_t'] >= 1020) & (filtered_df['arrival_t'] <= 1140)].index.tolist()
            allocatable_time_index = allocatable_index_1 + allocatable_index_2

            allocatable_short_activity_index = filtered_df[filtered_df['activity_t'] <= 30].index.tolist()
            allocatable_long_activity_index = filtered_df[
                (filtered_df['activity_t'] > 30) & (filtered_df['activity_t'] <= 150)].index.tolist()
            allocatable_short_index = list(set(allocatable_time_index) & set(allocatable_short_activity_index))
            allocatable_long_index = list(set(allocatable_time_index) & set(allocatable_long_activity_index))
            no_allocatable_index = list(set(park_index) - set(allocatable_short_index) - set(allocatable_long_index))

            filtered_df_adj = req_info.loc[adj_park_index]
            allocatable_index_1_adj = filtered_df_adj[
                (filtered_df_adj['arrival_t'] >= 480) & (filtered_df_adj['arrival_t'] <= 600)].index.tolist()
            allocatable_index_2_adj = filtered_df_adj[
                (filtered_df_adj['arrival_t'] >= 1020) & (filtered_df_adj['arrival_t'] <= 1140)].index.tolist()
            allocatable_time_index_adj = allocatable_index_1_adj + allocatable_index_2_adj

            allocatable_short_activity_index_adj = filtered_df_adj[filtered_df_adj['activity_t'] <= 30].index.tolist()
            allocatable_long_activity_index_adj = filtered_df_adj[
                (filtered_df_adj['activity_t'] > 30) & (filtered_df_adj['activity_t'] <= 150)].index.tolist()
            allocatable_short_index_adj = list(
                set(allocatable_time_index_adj) & set(allocatable_short_activity_index_adj))
            allocatable_long_index_adj = list(
                set(allocatable_time_index_adj) & set(allocatable_long_activity_index_adj))
            no_allocatable_index_adj = list(
                set(adj_park_index) - set(allocatable_short_index_adj) - set(allocatable_long_index_adj))

            try:
                # 短时停车请求分配到快充电桩
                m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in allocatable_short_index)
                m.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in allocatable_short_index_adj)
            except:
                pass
            try:
                # 长时停车请求分配到慢充桩
                m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in allocatable_long_index)
                m.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in allocatable_long_index_adj)
            except:
                pass
            # 其他请求分配到OPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in no_allocatable_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in no_allocatable_index_adj)

        # 快充电请求只能分配到快充CPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) <= 1 for m in fast_charge_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)

        # m.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 1 for a in adj_fast_charge_index)
        m.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in adj_fast_charge_index)
        m.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_fast_charge_index)

        # 慢充电请求只能分配到慢充CPS
        # m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) <= 1 for m in slow_charge_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        # m.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 1 for a in adj_slow_charge_index)
        m.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in adj_slow_charge_index)
        m.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_slow_charge_index)

        m.optimize()

        # gc.collect()

        obj4 = quicksum(
            cruise_t(z, S_NK[ith], EACH_INDEX[z], req_info['arrival_t'].loc[m]) * y_mz[(m, z)] for m in total_index for
            z in range(Z)) + need_adjust * quicksum(
            cruise_t(z, S_NK[ith], EACH_INDEX[z], req_info['arrival_t'].loc[a]) * y_az[(a, z)] for a in adj_total_index
            for z in range(Z))

        REVENUE_total[ith + 1][0] = obj.getValue()  # 目标函数
        REVENUE_total[ith + 1][1] = obj1.getValue()  # 停车场收益
        REVENUE_total[ith + 1][2] = park_revenue.getValue()  # 停车收益
        REVENUE_total[ith + 1][3] = char_revenue.getValue()  # 充电收益
        REVENUE_total[ith + 1][4] = obj2.getValue()  # 拒绝总数量
        REVENUE_total[ith + 1][5] = refuse_park.getValue()  # 停车拒绝数
        REVENUE_total[ith + 1][6] = refuse_char.getValue()  # 充电拒绝数
        REVENUE_total[ith + 1][7] = obj3.getValue()  # 行程时间
        REVENUE_total[ith + 1][8] = obj4.getValue()  # 巡游时间

        if need_adjust:

            assign_index = []

            for n in range(N):
                for temp_index, m_ in enumerate(total_index):
                    X_NM[n][temp_index] = x_nm[(n, m_)].X
                    if x_nm[(n, m_)].X == 1:
                        assign_index.append(m_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_zm[(z, m_)].X == 1:
                                assign_info.append([m_, n, z, REVENUE_total[ith + 1][8], i])
                for temp_index_, a_ in enumerate(adj_total_index):
                    X_NA[n][temp_index_] = x_na[(n, a_)].X
                    if x_na[(n, a_)].X == 1:
                        assign_index.append(a_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_za[(z, a_)].X == 1:
                                assign_info.append([a_, n, z, REVENUE_total[ith + 1][8], i])

            S_NK[ith + 1] = S_NK[ith] + np.matmul(X_NM, R_MK) + np.matmul(X_NA, R_AK)

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

            # 可以调整的请求在上一轮的分配结果
            X_NM_LAST_ = np.zeros((N, len(this_adj_total_index))).astype(int)
            X_NA_LAST_ = np.zeros((N, len(secondary_adj_total_index))).astype(int)
            for n in range(N):
                for x_nm_index, m__ in enumerate(this_adj_total_index):
                    X_NM_LAST_[n][x_nm_index] = x_nm[(n, m__)].X
                for x_na_index, a__ in enumerate(secondary_adj_total_index):
                    X_NA_LAST_[n][x_na_index] = x_na[(n, a__)].X
            X_NA_LAST_ = np.concatenate((X_NA_LAST_, X_NM_LAST_), axis=1).astype(int)

        else:
            assign_index = []

            for n in range(N):
                for temp_index, m_ in enumerate(total_index):
                    X_NM[n][temp_index] = x_nm[(n, m_)].X
                    if x_nm[(n, m_)].X == 1:
                        assign_index.append(m_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_zm[(z, m_)].X == 1:
                                assign_info.append([m_, n, z, REVENUE_total[ith + 1][8], i])
            S_NK[ith + 1] = S_NK[ith] + np.matmul(X_NM, R_MK)

            # 本轮可以调整的请求索引
            adj_total_index = list(set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            adj_park_index = list(set(adj_total_index) & set(park_index))
            adj_charge_index = list(set(adj_total_index) & set(charge_index))
            adj_fast_charge_index = list(set(adj_charge_index) & set(fast_charge_index))
            adj_slow_charge_index = list(set(adj_charge_index) & set(slow_charge_index))

            # 可以调整的请求在上一轮的分配结果
            X_NA_LAST_ = np.zeros((N, len(adj_total_index))).astype(int)
            for n in range(N):
                for temp_index_, m__ in enumerate(adj_total_index):
                    X_NA_LAST_[n][temp_index_] = x_nm[(n, m__)].X

        need_adjust = True

        # if len(adj_total_index) > 0:
        #     need_adjust = True
        # else:
        #     need_adjust = False

    # 计算收益
    # 3. 对多次分配的请求删除合并，计算收益
    dpr_assign = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'cruising_t', 'assign_t'],
                              data=np.array(assign_info).reshape((-1, 5)))
    dpr_assign_filter = dpr_assign.loc[dpr_assign.groupby('req_id')['assign_t'].idxmax()]
    # revenue = {'req_id': dpr_assign_filter.req_id, 'revenue': req_info['revenue'].loc[dpr_assign_filter.index.tolist()]}
    # revenue_df = pd.DataFrame(revenue)
    dpr_assign_filter_merge = pd.merge(dpr_assign_filter, req_info[['req_id', 'revenue']], on='req_id', how='left')
    travel_cost = []
    for i in range(len(dpr_assign_filter)):
        temp_o = req_info['O'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_d = req_info['D'].loc[dpr_assign_filter['req_id'].iloc[i]]
        temp_pl = dpr_assign_filter['pl_num'].iloc[i]
        travel_cost.append(cost_matrix[int(temp_pl)][int(temp_o)] + 2 * cost_matrix[int(temp_pl)][int(temp_d) + 2])
    dpr_assign_filter_merge['travel_cost'] = travel_cost
    # 5. 计算各类收益
    charge_index = req_info[req_info['label'] == 1].index.tolist()
    park_index = req_info[req_info['label'] == 0].index.tolist()
    charge_revenue = dpr_assign_filter_merge[dpr_assign_filter_merge['req_id'].isin(charge_index)]['revenue'].sum()
    park_revenue = dpr_assign_filter_merge[dpr_assign_filter_merge['req_id'].isin(park_index)]['revenue'].sum()
    average_travel_cost = dpr_assign_filter_merge['travel_cost'].sum() / len(dpr_assign_filter_merge)
    average_cruising_cost = dpr_assign_filter_merge['cruising_t'].sum() / len(dpr_assign_filter_merge)

    P_rev = charge_revenue + park_revenue  # 平台收入
    P_park_rev = park_revenue  # 停车收入
    P_char_rev = charge_revenue  # 充电收入
    P_avg_travel = average_travel_cost  # 平均行程时间
    P_avg_cruising = average_cruising_cost

    P_obj = 0
    P_refuse = 0
    P_park_refuse = 0
    P_char_refuse = 0

    for each in REVENUE_total:
        P_obj += each[0]  # 目标函数
        P_refuse += each[4]  # 拒绝数量
        P_park_refuse += each[5]  # 停车拒绝数量
        P_char_refuse += each[6]  # 充电拒绝数量
    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[len(I)][OPS_INDEX][:, :1440]) / (sum(pl[l].ordinary_num for l in range(Z)) * 1440)
    charging_util = np.sum(S_NK[len(I)][CPS_INDEX][:, :1440]) / (sum(pl[l].charge_num for l in range(Z)) * 1440)

    # acceptance rate of reservation requests
    req_num = len(r_mk)
    P_acc = (req_num - P_refuse) / req_num
    P_park_acc = (park_arrival_num - P_park_refuse) / park_arrival_num
    P_char_acc = (park_arrival_num * charge_ratio - P_char_refuse) / (park_arrival_num * charge_ratio)

    print("request number:" + str(req_num))
    print("objective value:" + str(P_obj))
    print("total revenue:" + str(P_rev))
    print("park revenue:" + str(P_park_rev))
    print("char revenue:" + str(P_char_rev))
    print("refuse number:" + str(P_refuse))
    print("refuse park number:" + str(P_park_refuse))
    print("refuse char number:" + str(P_char_refuse))
    print("acceptance rate of reservation requests:" + str(P_acc))
    print("acceptance rate of park reservation requests:" + str(P_park_acc))
    print("acceptance rate of char reservation requests:" + str(P_char_acc))
    print("parking utilization:" + str(parking_util))
    print("charging utilization:" + str(charging_util))
    print("travel cost:" + str(P_avg_travel))
    print("cruising cost:" + str(P_avg_cruising))

    result_list = [alpha1, alpha2, alpha3, req_num, P_obj, P_rev, P_park_rev, P_char_rev, P_refuse, P_park_refuse,
                   P_char_refuse, P_acc,
                   P_park_acc, P_char_acc, parking_util, charging_util, P_avg_travel, P_avg_cruising]

    # 保存数据
    os.chdir(r'G:\2023-纵向\停车分配\save_data\需求11供给11')

    try:
        os.makedirs(f'{park_arrival_num}-{charge_ratio}')
    except:
        pass
    os.chdir(f'{park_arrival_num}-{charge_ratio}')

    # 创建子文件夹
    folder_name = ['assign_info', 'basic_info', 'result_info', 'revenue_info', 'SNK_info']
    for each_folder in folder_name:
        try:
            os.makedirs(each_folder)
        except:
            pass

    os.chdir('basic_info')
    # 打开文件进行写入
    with open(f'rdp_{rule}.txt', 'w') as file:
        # 遍历列表，写入每个元素到文件
        for item in result_list:
            file.write(str(item) + '\n')  # 添加换行符以确保列表中的每个项目都在新的一行
    os.chdir('../')
    np.save(f"SNK_info/rdp_{rule}.npy", S_NK[-1])
    np.save(f"revenue_info/rdp_{rule}.npy", REVENUE_total)
    np.save(f"result_info/rdp_{rule}.npy", result_list)
    assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'cruising_t', 'assign_t'],
                                    data=np.array(assign_info).reshape((-1, 5)))
    assign_info_data.to_csv(f"assign_info/rdp_{rule}.csv", index=False)


if __name__ == '__main__':
    # 需求信息
    global park_arrival_num
    charge_ratio = 1

    # OD及停车场信息
    O, D, Z, cost_matrix = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
    pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z)
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

    for i in [100,150,200]:
        # 需求信息
        park_arrival_num = i
        req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\{park_arrival_num}-{charge_ratio}.csv")

        # 将请求排序 并计算所在的间隔
        decision_interval = 15
        req_info = req_info.sort_values(by="request_t")
        earliest_request = min(req_info['request_t'])
        req_info['request_interval'] = (req_info['request_t'] - earliest_request) // decision_interval
        request_interval = req_info['request_interval'].unique()

        # 计算达到到的时间间隔
        req_info['arrival_interval'] = (req_info['arrival_t'] - earliest_request) // decision_interval
        req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']

        park_fee = pl1.park_fee / 2  # 半个小时的费用
        charge_fee = [0, pl1.fast_charge_fee, pl1.slow_charge_fee]  # 每分钟的价格
        reserved_fee = pl1.reserve_fee  # 预约费用
        req_info['revenue'] = [(np.floor(req_info['activity_t'].iloc[i] / 15) * park_fee + (
                req_info['activity_t'].iloc[i] * (charge_fee[req_info['new_label'].iloc[i]])) + reserved_fee) for i in
                               range(len(req_info))]
        #
        N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数
        K = max(req_info['leave_t'])  # 时间长度
        I = request_interval  # 迭代次数  （需求个数）
        S_NK = np.zeros((len(I) + 1, N, K)).astype(int)  # 供给状态 每轮迭代更新
        REVENUE_total = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(len(I) + 1)]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；

        for j in range(1, 6):
            rdp(rule=j)
            print(f"park_num:{i},rule:{j}")