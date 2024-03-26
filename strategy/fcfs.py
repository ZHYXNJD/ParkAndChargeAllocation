"""
先预约先服务:
先发送预约请求的先处理
"""

from datetime import datetime
import pandas as pd
from entity import demand, parkinglot
from entity import OD
from utils import *
from gurobipy import *


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


# 按照预约顺序对请求排序
decision_interval = 15
req_info = req_info.sort_values(by='request_t')
earliest_request = min(req_info['request_t'])
req_info['request_interval'] = (req_info['request_t']-earliest_request) // decision_interval

#
N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数
K = max(req_info['leave_t'])  # 时间长度
I = len(req_info)  # 迭代次数  （需求个数）
S_NK = np.zeros((I + 1, N, K))  # 供给状态 每轮迭代更新
REVENUE_total = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(I + 1)]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；


# 泊位索引
T_ZN = T_ZN(Z, N, pl)
P_ZN, OPS_INDEX = P_ZN(Z, N, pl)
C_ZN, CPS_INDEX = C_ZN(Z, N, pl)
Fast_ZN,FAST_INDEX = Fast_ZN(Z, N, pl)
Slow_ZN,SLOW_INDEX = Slow_ZN(Z, N, pl)


def fcfs(rule):
    r_mk = get_rmk(req_info).astype(int)  # 二维的0-1矩阵  req_num,max_leave
    assign_info = []
    # np.save('r_mk.npy',r_mk)

    # 开始分配
    for i in range(I):
        print("-------------------第" + str(i) + "次分配结果-------------------")

        m = Model("fcfs")
        m.Params.OutputFlag = 1

        # 根据请求类型分配到普通泊位/充电桩
        label = req_info['label'].iloc[i]
        fast_label = 0
        if label:
            if req_info['activity_t'].iloc[i] <= 60:
                fast_label = 1

        X_NM = np.zeros((N, 1))  # 该请求的分配情况
        R_MK = r_mk[i].reshape((1, K))  # 该请求

        x_nm = m.addVars(N, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_zm = m.addVars(Z, vtype=GRB.BINARY, name="y_zm")  # z*m  停车场和泊位的从属关系
        y_mz = m.addVars(Z, vtype=GRB.BINARY, name="y_mz")  # m*z

        m.update()

        # 停车/充电的收益
        obj1 = pl1.reserve_fee * quicksum(x_nm[n] for n in range(N)) + \
               pl1.park_fee / 60 * quicksum(x_nm[n] * r_mk[i][k] for n in range(N) for k in range(K)) + \
               label * pl1.charge_fee * quicksum(x_nm[n] * r_mk[i][k] for n in range(N) for k in range(K))
        # 拒绝的数量
        obj2 = 1 - quicksum(x_nm[n] for n in range(N))
        # 行程时间 (起点到停车场+停车场到终点*2+巡游时间)
        o = req_info['O'].iloc[i]
        d = req_info['D'].iloc[i] + 2  # 加2是索引的缘故
        obj3 = quicksum(
            cost_matrix[z][o] * y_mz[z] + 2 * cost_matrix[z][d] * y_mz[z] for z in range(Z))

        # 总目标
        # 先到先服务只考虑行程时间
        obj = obj3 + 10000 * obj2  # 拒绝后会给很大惩罚
        m.setObjective(obj, GRB.MINIMIZE)
        m.addConstrs(y_zm[z] == y_mz[z] for z in range(Z))
        if not label:
            m.addConstrs(S_NK[i][n][k] + x_nm[n] * r_mk[i][k] <= 1 for n in OPS_INDEX for k in range(K))
        else:
            if fast_label:
                m.addConstrs(S_NK[i][n][k] + x_nm[n] * r_mk[i][k] <= 1 for n in FAST_INDEX for k in range(K))
            else:
                m.addConstrs(S_NK[i][n][k] + x_nm[n] * r_mk[i][k] <= 1 for n in SLOW_INDEX for k in range(K))
        m.addConstrs(quicksum(T_ZN[z][n] * x_nm[n] for n in range(N)) == y_zm[z] for z in range(Z))
        m.addConstr(quicksum(x_nm[n] for n in range(N)) <= 1)

        if label == 0:  # 停车请求
            if rule == 1:
                # 停车请求只能分配到OPS
                m.addConstr(quicksum(x_nm[n] for n in OPS_INDEX) <= 1)
                m.addConstr(quicksum(x_nm[n] for n in CPS_INDEX) == 0)
            else:
                # 停车请求可以分配到充电桩
                m.addConstr(quicksum(x_nm[n] for n in range(N)) <= 1)
        else:
            # 根据充电时间判断是快充还是慢充
            if fast_label:  # 快充
                # 快充请求只能分配到fast
                m.addConstr(quicksum(x_nm[n] for n in FAST_INDEX) <= 1)
                m.addConstr(quicksum(x_nm[n] for n in SLOW_INDEX) == 0)
                m.addConstr(quicksum(x_nm[n] for n in OPS_INDEX) == 0)

            else:  # 慢充
                # 慢充请求只能分配到slow
                m.addConstr(quicksum(x_nm[n] for n in FAST_INDEX) == 0)
                m.addConstr(quicksum(x_nm[n] for n in OPS_INDEX) == 0)
                m.addConstr(quicksum(x_nm[n] for n in SLOW_INDEX) <= 1)

        m.optimize()

        REVENUE_total[i + 1][0] = obj.getValue()  # 目标函数
        REVENUE_total[i + 1][1] = obj1.getValue()  # 停车场收益
        if label == 0:
            REVENUE_total[i + 1][2] = obj1.getValue()  # 停车收益
        else:
            REVENUE_total[i + 1][3] = obj1.getValue()  # 充电收益
        REVENUE_total[i + 1][4] = obj2.getValue()  # 拒绝总数量
        if label == 0:
            REVENUE_total[i + 1][5] = obj2.getValue()  # 停车拒绝数
        else:
            REVENUE_total[i + 1][6] = obj2.getValue()  # 充电拒绝数
        REVENUE_total[i + 1][7] = obj3.getValue()  # 行程时间

        for n in range(N):
            X_NM[n] = x_nm[n].X
            if x_nm[n].X == 1:
                for z in range(Z):
                    if y_zm[z].X == 1:
                        assign_info.append([req_info.iloc[i,0], n, z, req_info['request_interval'].iloc[i]])

        # S_NK 可以保存
        S_NK[i + 1] = S_NK[i] + np.matmul(X_NM, R_MK)

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
    parking_util = np.sum(S_NK[I][OPS_INDEX][:, :1440]) / (sum(pl[i].ordinary_num for i in range(Z)) * 1440)
    charging_util = np.sum(S_NK[I][CPS_INDEX][:, :1440]) / (sum(pl[i].charge_num for i in range(Z)) * 1440)

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

    result_list = [0,1,0,req_num, P_obj, P_rev, P_park_rev, P_char_rev, P_refuse, P_park_refuse, P_char_refuse, P_acc,
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
    with open('fcfs_{}.txt'.format(rule), 'w') as file:
        # 遍历列表，写入每个元素到文件
        for item in result_list:
            file.write(str(item)+'\n')  # 添加换行符以确保列表中的每个项目都在新的一行
    np.save("S_NK_total.npy", S_NK)
    np.save("REVENUE_total.npy", REVENUE_total)
    np.save("result_list.npy", result_list)
    assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],data=np.array(assign_info).reshape((-1, 4)))
    assign_info_data.to_csv('assign_info.csv', index=False)


fcfs(rule=1)
