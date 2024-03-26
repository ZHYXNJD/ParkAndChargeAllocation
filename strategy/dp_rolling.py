"""
先预约先服务的拓展
预约后在未到达之前可以调整 必须确保在下个决策间隔有泊位
决策间隔15min
"""
from datetime import datetime
from entity import OD,parkinglot,demand
from utils import *
from gurobipy import *
import pandas as pd


# 需求信息
park_arrival_num = 100
charge_ratio = 0.1
# req_info = demand.main(park_arrival_num=park_arrival_num,
#                        charge_ratio=charge_ratio)  # 返回信息 [request_t, arrival_t,activity_t,leave_t,label,O,D]
req_info = pd.read_csv("100-0.1-so.csv")

# OD及停车场信息
O, D, Z, cost_matrix = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z)
pl = [pl1, pl2, pl3, pl4]

# 将请求排序 并计算所在的间隔
decision_interval = 15
req_info = req_info.sort_values(by="request_t")
earliest_request = min(req_info['request_t'])
req_info['request_interval'] = (req_info['request_t']-earliest_request) // decision_interval
request_interval = req_info['request_interval'].unique()

# 计算达到到的时间间隔
req_info['arrival_interval'] = (req_info['arrival_t']-earliest_request)//decision_interval
req_info['diff'] = req_info['arrival_interval'] - req_info['request_interval']

#
N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数
K = max(req_info['leave_t'])  # 时间长度
I = request_interval  # 迭代次数  （需求个数）
S_NK = np.zeros((len(I) + 1, N, K)).astype(int)  # 供给状态 每轮迭代更新
REVENUE_total = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(len(I) + 1)]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；

# 泊位索引
T_ZN = T_ZN(Z, N, pl)
_, OPS_INDEX = P_ZN(Z, N, pl)
_, CPS_INDEX = C_ZN(Z, N, pl)
_,FAST_INDEX = Fast_ZN(Z, N, pl)
_,SLOW_INDEX = Slow_ZN(Z, N, pl)


def dp_rolling(rule,alpha1=0.7,alpha2=0.1,alpha3=0.2):

    r_mk = get_rmk_(req_info).astype(int)  # 注意区别于get_rmk
    # 存储请求id 泊位 停车场 分配时所在的间隔 (req_id space_num pl_num assign_t)
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
    for ith,i in enumerate(request_interval):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        m = Model("dp_rolling")
        m.Params.OutputFlag = 1

        # 本轮的请求
        this_req_info = req_info[req_info['request_interval'] == i]
        total_index = this_req_info.index.tolist()
        park_index = this_req_info[this_req_info['label'] == 0].index.tolist()
        charge_index = this_req_info[this_req_info['label'] == 1].index.tolist()
        fast_charge_index = list(set(this_req_info[this_req_info['activity_t'] <= 60].index.tolist()) & set(charge_index))
        slow_charge_index = list(set(this_req_info[this_req_info['activity_t'] >= 60].index.tolist()) & set(charge_index))

        X_NM = np.zeros((N, len(this_req_info))).astype(int)  # 该请求的分配情况
        R_MK = r_mk[total_index].reshape((len(this_req_info), K))  # 该请求
        X_NA_LAST = X_NA_LAST_

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
        park_revenue = pl1.reserve_fee * quicksum(x_nm[(n, m)] for n in range(N) for m in park_index) + \
                       pl1.park_fee / 60 * quicksum(x_nm[(n, m)] * r_mk[m][k] for n in range(N) for m in park_index for k in range(K)) + need_adjust*(
                       pl1.reserve_fee * quicksum(x_na[(n,a)] for n in range(N) for a in adj_park_index) +
                       pl1.park_fee / 60 * quicksum(x_na[(n,a)]*r_mk[a][k] for n in range(N) for a in adj_park_index for k in range(K)))

        char_revenue = pl1.reserve_fee * quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index) + \
                       (pl1.park_fee / 60 + pl1.charge_fee) * quicksum(x_nm[(n, m)] * r_mk[m][k] for n in range(N) for m in charge_index for k in range(K)) + need_adjust*(
                        pl1.reserve_fee * quicksum(x_na[(n, a)] for n in range(N) for a in adj_charge_index) +
                       (pl1.park_fee / 60 + pl1.charge_fee) * quicksum(x_na[(n, a)] * r_mk[a][k] for n in range(N) for a in adj_charge_index for k in range(K)))


        obj1 = park_revenue + char_revenue

        # 拒绝数量
        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in park_index)

        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index)

        obj2 = refuse_park + refuse_char

        # 行程时间
        obj3 = quicksum(cost_matrix[z][req_info['O'].loc[m]]*y_mz[(m,z)] + 2*cost_matrix[z][req_info['D'].loc[m]+2]*y_mz[(m,z)]for m in total_index for z in range(Z)) + \
               need_adjust*quicksum(cost_matrix[z][req_info['O'].loc[a]] * y_az[(a, z)] + 2 * cost_matrix[z][req_info['D'].loc[a] + 2] * y_az[(a, z)] for a in adj_total_index for z in range(Z))

        # 总目标
        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3
        # obj = alpha2 * obj2

        m.setObjective(obj, GRB.MAXIMIZE)

        if need_adjust:
            m.addConstrs(y_zm[(z, m)] == y_mz[(m, z)] for z in range(Z) for m in total_index)
            if X_NA_LAST.shape[1] != 0:
                m.addConstrs(y_za[(z, a)] == y_az[(a, z)] for z in range(Z) for a in adj_total_index)
                S_NK[ith] = S_NK[ith] - np.matmul(X_NA_LAST,R_AK)
                m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in park_index) + quicksum(x_na[(n, a)] * r_mk[a][k] for a in adj_park_index) <= 1 for n in OPS_INDEX for k in range(K))
                m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in fast_charge_index) + quicksum(x_na[(n, a)] * r_mk[a][k] for a in adj_fast_charge_index) <= 1 for n in FAST_INDEX for k in range(K))
                m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in slow_charge_index) + quicksum(x_na[(n, a)] * r_mk[a][k] for a in adj_slow_charge_index) <= 1 for n in SLOW_INDEX for k in range(K))
            else:
                m.addConstrs(S_NK[ith][n][k] + quicksum(
                    x_nm[(n, m)] * r_mk[m][k] for m in park_index) + quicksum(
                    x_na[(n, a)] * r_mk[a][k] for a in adj_park_index) <= 1 for n in OPS_INDEX for k in range(K))
                m.addConstrs(S_NK[ith][n][k] + quicksum(
                    x_nm[(n, m)] * r_mk[m][k] for m in fast_charge_index) + quicksum(
                    x_na[(n, a)] * r_mk[a][k] for a in adj_fast_charge_index) <= 1 for n in FAST_INDEX for k in
                             range(K))
                m.addConstrs(S_NK[ith][n][k] + quicksum(
                    x_nm[(n, m)] * r_mk[m][k] for m in slow_charge_index) + quicksum(
                    x_na[(n, a)] * r_mk[a][k] for a in adj_slow_charge_index) <= 1 for n in SLOW_INDEX for k in
                             range(K))
            m.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_zm[(z, m)] for z in range(Z) for m in total_index)
            m.addConstrs(quicksum(T_ZN[z][n] * x_na[(n, a)] for n in range(N)) == y_za[(z, a)] for z in range(Z) for a in adj_total_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in range(N)) == 1 for a in adj_total_index)
        else:
            m.addConstrs(y_zm[(z, m)] == y_mz[(m, z)] for z in range(Z) for m in total_index)
            m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in park_index) <= 1 for n in OPS_INDEX for k in range(K))
            m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in fast_charge_index) <= 1 for n in FAST_INDEX for k in range(K))
            m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in slow_charge_index) <= 1 for n in SLOW_INDEX for k in range(K))
            m.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_zm[(z, m)] for z in range(Z) for m in total_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

        if rule == 1:
            # rule1
            # 停车请求只能分配到OPS
            if need_adjust:
                m.addConstrs(x_nm[(n, m)] + x_na[(n, a)] <= 1 for n in OPS_INDEX for m in park_index for a in adj_park_index)
                m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
                m.addConstrs(quicksum(x_na[(n, a)] for n in CPS_INDEX) == 0 for a in adj_park_index)
            else:
                m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in park_index)
                m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
        else:
            # rule2:
            # 停车请求可以分配到OPS和CPS
            if need_adjust:
                m.addConstrs(quicksum(x_nm[(n, m)] + x_na[(n, a)]) <= 1 for n in range(N) for m in park_index for a in adj_park_index)
            else:
                m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in park_index)

        if need_adjust:
            # 快充电请求只能分配到快充CPS
            m.addConstrs(x_nm[(n, m)] + x_na[(n, a)] <= 1 for n in FAST_INDEX for m in fast_charge_index for a in adj_fast_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in SLOW_INDEX) == 0 for a in adj_fast_charge_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_fast_charge_index)
            # 慢充电请求只能分配到慢充CPS
            m.addConstrs(x_nm[(n, m)] + x_na[(n, a)] <= 1 for n in SLOW_INDEX for m in slow_charge_index for a in adj_slow_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in FAST_INDEX) == 0 for a in adj_slow_charge_index)
            m.addConstrs(quicksum(x_na[(n, a)] for n in OPS_INDEX) == 0 for a in adj_slow_charge_index)
        else:
            # 快充电请求只能分配到快充CPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) <= 1 for m in fast_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) == 0 for m in fast_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in fast_charge_index)
            # 慢充电请求只能分配到慢充CPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in SLOW_INDEX) <= 1 for m in slow_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in FAST_INDEX) == 0 for m in slow_charge_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) == 0 for m in slow_charge_index)

        m.optimize()

        gc.collect()

        REVENUE_total[ith + 1][0] = obj.getValue()  # 目标函数
        REVENUE_total[ith + 1][1] = obj1.getValue()  # 停车场收益
        REVENUE_total[ith + 1][2] = park_revenue.getValue()  # 停车收益
        REVENUE_total[ith + 1][3] = char_revenue.getValue()  # 充电收益
        REVENUE_total[ith + 1][4] = obj2.getValue()  # 拒绝总数量
        REVENUE_total[ith + 1][5] = refuse_park.getValue()  # 停车拒绝数
        REVENUE_total[ith + 1][6] = refuse_char.getValue()  # 充电拒绝数
        REVENUE_total[ith + 1][7] = obj3.getValue()  # 行程时间

        if need_adjust:

            assign_index = []

            for n in range(N):
                for temp_index, m_ in enumerate(total_index):
                    X_NM[n][temp_index] = x_nm[(n, m_)].X
                    if x_nm[(n, m_)].X == 1:
                        assign_index.append(m_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_zm[(z,m_)].X == 1:
                                assign_info.append([m_, n, z, i])
                for temp_index_,a_ in enumerate(adj_total_index):
                    X_NA[n][temp_index_] = x_na[(n,a_)].X
                    if x_na[(n,a_)].X ==1:
                        assign_index.append(a_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_za[(z,a_)].X == 1:
                                assign_info.append([a_, n, z, i])

            S_NK[ith + 1] = S_NK[ith] + np.matmul(X_NM, R_MK) + np.matmul(X_NA,R_AK)

            # 存储分配的信息

            # 将上一轮可以调整的请求和本轮可以调整的请求合并 继续调整
            # 本轮可以调整的请求索引
            this_adj_total_index = list(set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
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
                for x_na_index,a__ in enumerate(secondary_adj_total_index):
                    X_NA_LAST_[n][x_na_index] = x_na[(n,a__)].X
            X_NA_LAST_ = np.concatenate((X_NA_LAST_,X_NM_LAST_),axis=1).astype(int)

        else:
            assign_index = []

            for n in range(N):
                for temp_index, m_ in enumerate(total_index):
                    X_NM[n][temp_index] = x_nm[(n, m_)].X
                    if x_nm[(n, m_)].X == 1:
                        assign_index.append(m_)  # 获得已经分配的请求索引
                        for z in range(Z):
                            if y_zm[(z,m_)].X == 1:
                                assign_info.append([m_, n, z, i])
            S_NK[ith + 1] = S_NK[ith] + np.matmul(X_NM, R_MK)

            # 本轮可以调整的请求索引
            adj_total_index = list(set(this_req_info[this_req_info['diff'] > 1].index.tolist()) & set(assign_index))
            adj_park_index = list(set(adj_total_index) & set(park_index))
            adj_charge_index = list(set(adj_total_index) & set(charge_index))
            adj_fast_charge_index = list(set(adj_charge_index) & set(fast_charge_index))
            adj_slow_charge_index = list(set(adj_charge_index) & set(slow_charge_index))

            # 可以调整的请求在上一轮的分配结果
            X_NA_LAST_ = np.zeros((N,len(adj_total_index))).astype(int)
            for n in range(N):
                for temp_index_, m__ in enumerate(adj_total_index):
                    X_NA_LAST_[n][temp_index_] = x_nm[(n, m__)].X

        need_adjust = True

        # if len(adj_total_index) > 0:
        #     need_adjust = True
        # else:
        #     need_adjust = False

    P_obj = 0
    P_rev = 0
    P_park_rev = 0
    P_char_rev = 0
    P_refuse = 0
    P_park_refuse = 0
    P_char_refuse = 0
    P_travel = 0
    for each in REVENUE_total:
        P_obj += each[0]  # 目标函数
        P_rev += each[1]  # 平台收入
        P_park_rev += each[2]  # 停车收入
        P_char_rev += each[3]  # 充电收入
        P_refuse += each[4]  # 拒绝数量
        P_park_refuse += each[5]  # 停车拒绝数量
        P_char_refuse += each[6]  # 充电拒绝数量
        P_travel += each[7]  # 行程时间（总），需要计算平均值

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[len(I)][OPS_INDEX][:, :1440]) / (sum(pl[l].ordinary_num for l in range(Z)) * 1440)
    charging_util = np.sum(S_NK[len(I)][CPS_INDEX][:, :1440]) / (sum(pl[l].charge_num for l in range(Z)) * 1440)

    # acceptance rate of reservation requests
    req_num = len(r_mk)
    P_acc = (req_num - P_refuse) / req_num
    P_park_acc = (park_arrival_num - P_park_refuse) / park_arrival_num
    P_char_acc = (park_arrival_num * charge_ratio - P_char_refuse) / (park_arrival_num * charge_ratio)

    # 平均行程时间
    P_travel = P_travel / (req_num - P_refuse)

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
    print("walk cost:" + str(P_travel))

    result_list = [alpha1,alpha2,alpha3,req_num, P_obj, P_rev,P_park_rev, P_char_rev, P_refuse, P_park_refuse, P_char_refuse, P_acc,
                   P_park_acc, P_char_acc, parking_util, charging_util, P_travel]

    # 保存数据
    os.chdir('../')
    try:
        os.makedirs('save_data')
    except:
        pass
    os.chdir('save_data')

    folder_name = str(datetime.now().strftime("%m-%d-%H-%M"))
    os.makedirs(folder_name)
    os.chdir(folder_name)
    # 打开文件进行写入
    with open('dp_rolling{}.txt'.format(rule), 'w') as file:
        # 遍历列表，写入每个元素到文件
        for item in result_list:
            file.write(str(item) + '\n')  # 添加换行符以确保列表中的每个项目都在新的一行
    np.save("S_NK_total.npy", S_NK)
    np.save("REVENUE_total.npy", REVENUE_total)
    np.save("result_list.npy", result_list)
    assign_info_data = pd.DataFrame(columns=['req_id','space_num','pl_num','assign_t'],data=np.array(assign_info).reshape((-1,4)))
    assign_info_data.to_csv('assign_info.csv',index=False)


dp_rolling(rule=1)
























