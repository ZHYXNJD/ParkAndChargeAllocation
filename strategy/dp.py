"""
动态优化
以15min为决策间隔
"""
from datetime import datetime
from entity import demand, parkinglot
from entity import OD
from utils import *
from gurobipy import *
import pandas as pd

# 需求信息
park_arrival_num = 100
charge_ratio = 0.1
# req_info = demand.main(park_arrival_num=park_arrival_num,
#                        charge_ratio=charge_ratio)  # 返回信息 [request_t, arrival_t,activity_t,leave_t,label,O,D]
req_info = pd.read_csv("100-0.1-so.csv",index=False)

# OD及停车场信息
O, D, Z, cost_matrix = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量 时间矩阵
pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z)
pl = [pl1, pl2, pl3, pl4]
# 计算收益
req_info['revenue'] = [(req_info['activity_t'].iloc[i] * (pl1.park_fee/60+pl1.charge_fee*req_info['label'].iloc[i]) + pl1.reserve_fee) for i in range(len(req_info))]


# 将请求排序 并计算所在的间隔
decision_interval = 15
req_info = req_info.sort_values(by="request_t")
earliest_request = min(req_info['request_t'])
req_info['request_interval'] = (req_info['request_t']-earliest_request) // decision_interval
request_interval = req_info['request_interval'].unique()

#
N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数
K = max(req_info['leave_t'])  # 时间长度
I = request_interval  # 迭代次数  （需求个数）
S_NK = np.zeros((len(I) + 1, N, K)).astype(int) # 供给状态 每轮迭代更新
X_NM_total = []  # 每轮分配结果
REVENUE_total = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(len(I) + 1)]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；

# 泊位索引
T_ZN = T_ZN(Z, N, pl)
P_ZN, OPS_INDEX = P_ZN(Z, N, pl)
C_ZN, CPS_INDEX = C_ZN(Z, N, pl)
Fast_ZN,FAST_INDEX = Fast_ZN(Z, N, pl)
Slow_ZN,SLOW_INDEX = Slow_ZN(Z, N, pl)


def dp(rule,alpha1=0.7,alpha2=0.1,alpha3=0.2):

    r_mk = get_rmk(req_info).astype(int)  # 二维的0-1矩阵  req_num,max_leave
    assign_info = []

    for ith, i in enumerate(request_interval):
        print("-------------------第" + str(ith) + "次分配结果-------------------")

        m = Model("dp")
        m.Params.OutputFlag = 1

        temp_req_info = req_info[req_info['request_interval']==i]
        total_index = temp_req_info.index.tolist()
        park_index = temp_req_info[temp_req_info['label'] == 0].index.tolist()
        charge_index = temp_req_info[temp_req_info['label'] == 1].index.tolist()
        fast_charge_index = list(set(temp_req_info[temp_req_info['activity_t'] <= 60].index.tolist()) & set(charge_index))
        slow_charge_index = list(set(temp_req_info[temp_req_info['activity_t'] >= 60].index.tolist()) & set(charge_index))

        X_NM = np.zeros((N,len(temp_req_info))).astype(int) # 该请求的分配情况
        R_MK = r_mk[total_index].reshape((len(temp_req_info), K))  # 该请求

        x_nm = m.addVars({(n, m) for n in range(N) for m in total_index}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_zm = m.addVars({(z, m) for z in range(Z) for m in total_index}, vtype=GRB.BINARY, name="y_zm")  # z*m
        y_mz = m.addVars({(m, z) for m in total_index for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        m.update()

        # 平台收益： 预约费用 + 停车费用 + 充电费用
        park_revenue = pl1.reserve_fee * quicksum(x_nm[(n, m)] for n in range(N) for m in park_index) + \
                       pl1.park_fee / 60 * quicksum(
            x_nm[(n, m)] * r_mk[m][k] for n in range(N) for m in park_index for k in range(K))

        char_revenue = pl1.reserve_fee * quicksum(x_nm[(n, m)] for n in range(N) for m in charge_index) + \
                       pl1.park_fee / 60 * quicksum(
            x_nm[(n, m)] * r_mk[m][k] for n in range(N) for m in charge_index for k in range(K)) + \
                       pl1.charge_fee * quicksum(
            x_nm[(n, m)] * r_mk[m][k] for n in range(N) for m in charge_index for k in
            range(K))

        obj1 = park_revenue + char_revenue

        # 平台拒绝数量
        obj2 = len(total_index) - quicksum(x_nm[(n, m)] for m in total_index for n in range(N))
        # 拒绝的停车数量
        refuse_park = len(park_index) - quicksum(x_nm[(n, m)] for m in park_index for n in range(N))
        refuse_char = len(charge_index) - quicksum(x_nm[(n, m)] for m in charge_index for n in range(N))

        # 行程时间  没有办法考虑巡游时间
        obj3 = quicksum(
            cost_matrix[z][temp_req_info['O'].loc[m]] * y_mz[(m, z)] + 2 * cost_matrix[z][temp_req_info['D'].loc[m] + 2] * y_mz[
                (m, z)]
            for m in total_index for z in range(Z))

        # 总目标
        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3

        m.setObjective(obj, GRB.MAXIMIZE)
        m.addConstrs(y_zm[(z, m)] == y_mz[(m, z)] for z in range(Z) for m in total_index)
        m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in park_index) <= 1 for n in OPS_INDEX for k in range(K))
        m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in fast_charge_index) <= 1 for n in FAST_INDEX for k in range(K))
        m.addConstrs(S_NK[ith][n][k] + quicksum(x_nm[(n, m)] * r_mk[m][k] for m in slow_charge_index) <= 1 for n in SLOW_INDEX for k in range(K))
        m.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n, m)] for n in range(N)) == y_zm[(z, m)] for z in range(Z) for m in total_index)
        m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in total_index)

        if rule == 1:
            # rule1
            # 停车请求只能分配到OPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in OPS_INDEX) <= 1 for m in park_index)
            m.addConstrs(quicksum(x_nm[(n, m)] for n in CPS_INDEX) == 0 for m in park_index)
        else:
            # rule2:
            # 停车请求可以分配到OPS和CPS
            m.addConstrs(quicksum(x_nm[(n, m)] for n in range(N)) <= 1 for m in park_index)

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

        for n in range(N):
            for temp_index, m_ in enumerate(total_index):
                X_NM[n][temp_index] = x_nm[(n, m_)].X
                if x_nm[(n,m_)].X == 1:
                    for z in range(Z):
                        if y_zm[(z, m_)].X == 1:
                            assign_info.append([m_, n, z, i])

        S_NK[ith + 1] = S_NK[ith] + np.matmul(X_NM, R_MK)


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

    result_list = [alpha1,alpha2,alpha3,req_num, P_obj, P_rev, P_park_rev, P_char_rev, P_refuse, P_park_refuse, P_char_refuse, P_acc,
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
    with open('dp_{}.txt'.format(rule), 'w') as file:
        # 遍历列表，写入每个元素到文件
        for item in result_list:
            file.write(str(item) + '\n')  # 添加换行符以确保列表中的每个项目都在新的一行
    np.save("S_NK_total.npy", S_NK)
    np.save("REVENUE_total.npy", REVENUE_total)
    np.save("result_list.npy", result_list)
    assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],data=np.array(assign_info).reshape((-1, 4)))
    assign_info_data.to_csv('assign_info.csv', index=False)

dp(rule=1)