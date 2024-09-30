"""
先预约先服务:
先发送预约请求的先处理
"""
import pandas as pd
from entity import OD
from utils import *
from gurobipy import *


def fcfs(rule, alpha1=1, alpha2=1, alpha3=1, alpha4=5):
    """
        alpha1:收益系数
        alpha2:时间系数
        alpha3:占有率差值系数
        alpha4:拒绝惩罚系数
        """
    r_mk = get_rmk(req_info).astype(int)  # 二维的0-1矩阵  req_num,max_leave
    assign_info = []

    # 开始分配
    for i in range(I):
        print("-------------------第" + str(i) + "次分配结果-------------------")

        m = Model("fcfs")
        m.Params.OutputFlag = 1

        # 根据请求类型分配到普通泊位/充电桩
        label = req_info['new_label'].iloc[i]
        arrival_t = req_info['arrival_t'].iloc[i]
        activity_t = req_info['activity_t'].iloc[i]

        X_NM = np.zeros((N, 1))  # 该请求的分配情况
        R_MK = r_mk[i].reshape((1, K))  # 该请求

        x_nm = m.addVars({(n,0) for n in range(N)}, vtype=GRB.BINARY, name="x_nm")  # n*m
        y_zm = m.addVars({(z,0) for z in range(Z)}, vtype=GRB.BINARY, name="y_zm")  # z*m  停车场和泊位的从属关系
        y_mz = m.addVars({(z,0) for z in range(Z)}, vtype=GRB.BINARY, name="y_mz")  # m*z

        m.update()

        # 停车/充电的收益
        obj1 = quicksum(x_nm[(n,0)] for n in range(N)) * req_info['std_revenue'].iloc[i]
        obj1_ = quicksum(x_nm[(n,0)] for n in range(N)) * req_info['revenue'].iloc[i]

        # 拒绝的数量
        obj4 = 1 - quicksum(x_nm[(n,0)] for n in range(N))

        # 行程时间 (起点到停车场+停车场到终点*2+巡游时间)
        o = req_info['O'].iloc[i]
        d = req_info['D'].iloc[i] + 2  # 加2是索引的缘故
        obj2 = quicksum(cost_matrix[z][o] * y_mz[(z,0)] + 2 * cost_matrix[z][d] * y_mz[(z,0)] for z in range(Z))
        obj2_ = quicksum(cost_matrix_[z][o] * y_mz[(z,0)] + 2 * cost_matrix_[z][d] * y_mz[(z,0)] for z in range(Z))

        # 停车场占有率
        obj3 = quicksum(y_mz[(z,0)] * pl_occ(z, S_NK[i], EACH_INDEX[z], req_info['arrival_t'].iloc[i]) for z in range(Z))

        # 总目标
        # 停车场收益 - 出行距离 - 停车场占有率
        obj = alpha1 * obj1 - alpha2 * obj2 - alpha3 * obj3 - alpha4 * obj4
        m.setObjective(obj, GRB.MAXIMIZE)
        m.addConstrs(y_zm[(z,0)] == y_mz[(z,0)] for z in range(Z))

        m.addConstrs(S_NK[i][n][k] + x_nm[(n,0)] * r_mk[i][k] <= 1 for n in range(N) for k in range(K))
        m.addConstrs(quicksum(T_ZN[z][n] * x_nm[(n,0)] for n in range(N)) == y_zm[(z,0)] for z in range(Z))
        m.addConstr(quicksum(x_nm[(n,0)] for n in range(N)) <= 1)

        if label == 0:  # 停车请求
            if rule == 1:
                # 停车请求只能分配到OPS
                m.addConstr(quicksum(x_nm[(n,0)] for n in CPS_INDEX) == 0)
            elif rule == 2:
                # 停车请求可以分配到充电桩
                pass
            elif rule == 3:
                # 只有在规定时间内停车请求可以分配到充电桩
                # 8-10 17-19 可以停放（到达时间为480-600或1020-1140）
                if (480 <= arrival_t <= 600) | (1020 <= arrival_t <= 1140):
                    # 停车请求可以分配到充电桩
                    pass
                else:
                    # 停车请求只能分配到OPS
                    m.addConstr(quicksum(x_nm[(n,0)] for n in CPS_INDEX) == 0)
            elif rule == 4:
                # 短时停车停放在快充桩 长时停车停放在慢充桩
                # 短时停车(<=30分钟) 长时停车(30-150)
                if activity_t <= 30:
                    # 短时停车请求分配到快充电桩
                    m.addConstr(quicksum(x_nm[(n,0)] for n in SLOW_INDEX) == 0)
                elif 30 < activity_t <= 150:
                    # 长时停车请求分配到慢充桩
                    m.addConstr(quicksum(x_nm[(n,0)] for n in FAST_INDEX) == 0)
                else:
                    # 停车请求只能分配到OPS
                    m.addConstr(quicksum(x_nm[(n,0)] for n in CPS_INDEX) == 0)
            else:
                # rule3和rule4的结合
                if (480 <= arrival_t <= 600) | (1020 <= arrival_t <= 1140):
                    if activity_t <= 30:
                        # 短时停车请求分配到快充电桩

                        m.addConstr(quicksum(x_nm[(n,0)] for n in SLOW_INDEX) == 0)
                    elif 30 < activity_t <= 150:
                        # 长时停车请求分配到慢充桩

                        m.addConstr(quicksum(x_nm[(n,0)] for n in FAST_INDEX) == 0)
                    else:
                        # 停车请求只能分配到OPS

                        m.addConstr(quicksum(x_nm[(n,0)] for n in CPS_INDEX) == 0)
                else:
                    # 停车请求只能分配到OPS

                    m.addConstr(quicksum(x_nm[(n,0)] for n in CPS_INDEX) == 0)

        elif label == 1:
            # 快充请求只能分配到fast

            m.addConstr(quicksum(x_nm[(n,0)] for n in SLOW_INDEX) == 0)
            m.addConstr(quicksum(x_nm[(n,0)] for n in OPS_INDEX) == 0)
        else:
            # 慢充
            # 慢充请求只能分配到slow
            m.addConstr(quicksum(x_nm[(n,0)] for n in FAST_INDEX) == 0)
            m.addConstr(quicksum(x_nm[(n,0)] for n in OPS_INDEX) == 0)

        m.optimize()

        REVENUE_total[i + 1][0] = obj.getValue()  # 目标函数
        REVENUE_total[i + 1][1] = obj1_.getValue()  # 停车场收益
        if label == 0:
            REVENUE_total[i + 1][2] = REVENUE_total[i + 1][1]  # 停车收益
        else:
            REVENUE_total[i + 1][3] = REVENUE_total[i + 1][1]  # 充电收益
        REVENUE_total[i + 1][4] = obj4.getValue()  # 拒绝总数量
        if label == 0:
            REVENUE_total[i + 1][5] = obj4.getValue()  # 停车拒绝数
        else:
            REVENUE_total[i + 1][6] = obj4.getValue()  # 充电拒绝数
        REVENUE_total[i + 1][7] = obj2_.getValue()  # 行程时间
        # REVENUE_total[i + 1][8] = obj4.getValue()  # 停车场占有率

        for n in range(N):
            flag = 0
            X_NM[n] = x_nm[(n,0)].X
            if x_nm[(n,0)].X == 1:
                for z in range(Z):
                    if y_zm[(z,0)].X == 1:
                        assign_info.append(
                                [req_info.iloc[i, 0], n, z, req_info['request_interval'].iloc[i]])
                        flag = 1
                        break
            if flag:
                break



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
    # P_cruising = 0

    for each in REVENUE_total:
        P_obj += each[0]  # 目标函数
        P_rev += each[1]  # 平台收入
        P_park_rev += each[2]  # 停车收入
        P_char_rev += each[3]  # 充电收入
        P_refuse += each[4]  # 拒绝数量
        P_park_refuse += each[5]  # 停车拒绝数量
        P_char_refuse += each[6]  # 充电拒绝数量
        P_travel += each[7]  # 行程时间
        # P_cruising += each[8]  # 巡游时间

    # parking utilization and charging utilization
    parking_util = np.sum(S_NK[-1][OPS_INDEX][:, :1440]) / (sum(pl[i].ordinary_num for i in range(Z)) * 1440)
    charging_util = np.sum(S_NK[-1][CPS_INDEX][:, :1440]) / (sum(pl[i].charge_num for i in range(Z)) * 1440)

    # acceptance rate of reservation requests
    req_num = len(r_mk)
    P_acc = (req_num - P_refuse) / req_num
    P_park_acc = (park_arrival_num - P_park_refuse) / park_arrival_num
    P_char_acc = (park_arrival_num * charge_ratio - P_char_refuse) / (park_arrival_num * charge_ratio)

    # 平均行程时间
    P_travel = P_travel / (req_num - P_refuse)
    # 平均巡游时间
    # P_cruising = P_cruising / (req_num - P_refuse)

    # 平均占有率差值
    temp_occ = np.array([np.sum(S_NK[-1, EACH_INDEX[z]], axis=0) / pl[z].total_num for z in range(Z)]).reshape(Z, K)
    P_occ_diff = np.mean(np.sum((temp_occ - np.mean(temp_occ, axis=0)) ** 2 / Z, axis=0))

    result_dict = {"alpha1":alpha1,"alpha2":alpha2,"alpha3":alpha3,"alpha4":alpha4,"request number": req_num, "objective value": P_obj, "total revenue": P_rev,
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
    folder_name = ['assign_info','result_info','revenue_info', 'SNK_info']
    for each_folder in folder_name:
        try:
            os.makedirs(each_folder)
        except:
            pass

    np.save(f"SNK_info/fbfs_{rule}.npy", S_NK[-1])
    np.save(f"revenue_info/fbfs_{rule}.npy", REVENUE_total)
    np.save(f"result_info/fbfs_{rule}.npy", result_dict)
    assign_info_data = pd.DataFrame(columns=['req_id', 'space_num', 'pl_num', 'assign_t'],
                                    data=np.array(assign_info).reshape((-1, 4)))
    assign_info_data.to_csv(f"assign_info/fbfs_{rule}.csv", index=False)


if __name__ == '__main__':
    # 需求信息
    global park_arrival_num
    charge_ratio = 1

    # OD及停车场信息
    O, D, Z = OD.OdCost().get_od_info()  # 起点数量 终点数量 停车场数量
    cost_matrix_ = OD.OdCost().cost_matrix  # 非标准化
    cost_matrix = OD.OdCost().get_std_cost()  # 标准化
    pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=Z,config="1:1")
    pl = [pl1, pl2, pl3, pl4]

    # 计算收益
    # 停车费用按半小时记，不足半小时的按半小时计算
    # 充电费用按每分钟计费
    park_fee = pl1.park_fee / 2  # 半个小时的费用
    charge_fee = [0, pl1.fast_charge_fee, pl1.slow_charge_fee]  # 每分钟的价格
    reserved_fee = pl1.reserve_fee  # 预约费用

    #
    N = pl1.total_num + pl2.total_num + pl3.total_num + pl4.total_num  # 总泊位数

    # 泊位索引
    T_ZN, _, EACH_INDEX = T_ZN(Z, N, pl)
    _, OPS_INDEX = P_ZN(Z, N, pl)
    _, CPS_INDEX = C_ZN(Z, N, pl)
    _, FAST_INDEX = Fast_ZN(Z, N, pl)
    _, SLOW_INDEX = Slow_ZN(Z, N, pl)

    for i in range(50,525,25):
        park_arrival_num = i
        req_info = pd.read_csv(f"G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\{park_arrival_num}-{charge_ratio}.csv")

        # 按照预约顺序对请求排序
        decision_interval = 15
        req_info.sort_values(by='request_t', inplace=True)
        earliest_request = min(req_info['request_t'])
        req_info['request_interval'] = (req_info['request_t'] - earliest_request) // decision_interval

        req_info['revenue'] = [(np.floor(req_info['activity_t'].iloc[i] / 15) * park_fee + (
                req_info['activity_t'].iloc[i] * (charge_fee[req_info['new_label'].iloc[i]])) + reserved_fee) for i in
                               range(len(req_info))]
        req_info['std_revenue'] = (req_info['revenue'] - min(req_info['revenue'])) / (
                max(req_info['revenue']) - min(req_info['revenue']))

        K = max(req_info['leave_t'])  # 时间长度
        I = len(req_info)  # 迭代次数  （需求个数）
        S_NK = np.zeros((I + 1, N, K))  # 供给状态 每轮迭代更新
        REVENUE_total = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(I + 1)]  # i时段 目标函数，停车/充电收益，拒绝数量，步行时间；

        # for j in range(2, 6):
        #     fcfs(rule=j)
        #     print(f"park_num:{i},rule:{j}")
        fcfs(rule=2)
