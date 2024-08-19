"""
强化学习环境搭建
"""
import numpy as np
import pandas as pd
from entity import parkinglot, OD, demand

pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot()
pl = [pl1, pl2, pl3, pl4]
pl_num = len(pl)

O, D, pl, cost_matrix = OD.OdCost().get_od_info()

ordinary_num = [pl1.ordinary_num, pl2.ordinary_num, pl3.ordinary_num, pl4.ordinary_num]
fast_charge_num = [pl1.fast_charge_space, pl2.fast_charge_space, pl3.fast_charge_space, pl4.fast_charge_space]
slow_charge_num = [pl1.slow_charge_space, pl2.slow_charge_space, pl3.slow_charge_space, pl4.slow_charge_space]


def get_revenue(req_info):
    park_fee = pl1.park_fee / 2  # 半个小时的费用
    charge_fee = [0, pl1.fast_charge_fee, pl1.slow_charge_fee]  # 每分钟的价格
    reserved_fee = pl1.reserve_fee  # 预约费用
    req_info['revenue'] = [(np.floor(req_info['activity_t'].iloc[i] / 15) * park_fee + (
            req_info['activity_t'].iloc[i] * (charge_fee[req_info['new_label'].iloc[i]])) + reserved_fee) for i in
                           range(len(req_info))]
    req_info['revenue_std'] = (req_info['revenue'] - min(req_info['revenue'])) / (
            max(req_info['revenue']) - min(req_info['revenue']))
    return req_info


def get_train_req(park_arrival_num, charge_ratio):
    # 更新需求
    # req_info = demand.main(park_arrival_num, charge_ratio, train=True)
    req_info = pd.read_csv(r"G:\2023-纵向\停车分配\需求分布\demand0607\400-0.25.csv")
    return get_revenue(req_info)


def match_action(x, y):
    if x == 0:
        if y == 0:
            return 1
        elif y == 1:
            return 2
        elif y == 2:
            return 3
    elif x == 1:
        if y == 0:
            return 4
        elif y == 1:
            return 5
        elif y == 2:
            return 6
    elif x == 2:
        if y == 0:
            return 7
        elif y == 1:
            return 8
        elif y == 2:
            return 9
    elif x == 3:
        if y == 0:
            return 10
        elif y == 1:
            return 11
        elif y == 2:
            return 12
    else:
        return 0


def get_so_assign(park_arrival_num, charge_ratio):
    # 更新需求
    # req_info = demand.main(park_arrival_num, charge_ratio, train=True)
    req_info = pd.read_csv(r"G:\2023-纵向\停车分配\需求分布\demand0607\400-0.25.csv")
    so_assign = pd.read_csv(r'G:\2023-纵向\停车分配\save_data\需求41供给11\400-0.25\assign_info\so_2.csv')
    so_assign = pd.merge(so_assign, req_info[['req_id', 'request_t', 'arrival_t', 'new_label']], on='req_id',
                         how='right')
    so_assign['opt_action'] = so_assign[['pl_num', 'new_label']].apply(lambda x: match_action(x.iloc[0], x.iloc[1]),
                                                                       axis=1)
    return so_assign


def get_eval_req():
    evaluate_req_info = pd.read_csv(r"G:\2023-纵向\停车分配\需求分布\demand0607\400-0.25.csv")
    return get_revenue(evaluate_req_info)


class ParkingLotManagement:
    def __init__(self, pl_id, req_information):
        self.req_info = None
        self.id = pl_id
        self.park_info = []
        self.add_type = []
        self.av_ops = ordinary_num[self.id]
        self.av_fcps = fast_charge_num[self.id]
        self.av_scps = slow_charge_num[self.id]
        self.total_num = self.av_ops + self.av_fcps + self.av_scps
        self.cruising_t = 0
        self.req_info = req_information

        """
        更新停车管理系统
        remaining_t:leave_t - current 
        剩余停放时间:离开时间-当前时间 
        """
        self.av_ops_status = np.zeros(
            (ordinary_num[self.id], 6))  # arrival_t  leave_t  type  revenue current_t remaining_t
        self.av_fcps_status = np.zeros(
            (fast_charge_num[self.id], 6))  # arrival_t  leave_t  type  revenue current_t remaining_t
        self.av_scps_status = np.zeros(
            (slow_charge_num[self.id], 6))  # arrival_t  leave_t  type  revenue current_t remaining_t

        self.init_ps_status = np.concatenate((self.av_ops_status, self.av_fcps_status, self.av_scps_status), axis=0)
        self.supply_status = None

    def add_req(self, req_id, action_type):
        self.park_info.append(req_id)
        self.add_type.append(action_type)
        add_type = action_type % 3
        if add_type == 1:

            available_index = list(np.where(self.av_ops_status[:, -1] == 0)[0])[0]

            self.av_ops_status[available_index, :4] = \
                self.req_info[['arrival_t', 'leave_t', 'new_label', 'revenue_std']].loc[req_id].values
            self.av_ops -= 1

        elif add_type == 2:

            available_index = list(np.where(self.av_fcps_status[:, -1] == 0)[0])[0]

            self.av_fcps_status[available_index, :4] = \
                self.req_info[['arrival_t', 'leave_t', 'new_label', 'revenue_std']].loc[req_id].values
            self.av_fcps -= 1

        else:

            available_index = list(np.where(self.av_scps_status[:, -1] == 0)[0])[0]

            self.av_scps_status[available_index, :4] = \
                self.req_info[['arrival_t', 'leave_t', 'new_label', 'revenue_std']].loc[req_id].values
            self.av_scps -= 1

    def update_supply_status(self, current_t):

        # 更新剩余停车时间
        self.av_ops_status[:, -2] = current_t
        self.av_ops_status[:, -1] = self.av_ops_status[:, 1] - current_t
        # remaining t < 0 重置改泊位状态
        available_p_index = list(np.where(self.av_ops_status[:, -1] <= 0)[0])
        self.av_ops_status[available_p_index] = 0

        self.av_fcps_status[:, -2] = current_t
        self.av_fcps_status[:, -1] = self.av_fcps_status[:, 1] - current_t
        available_fc_index = list(np.where(self.av_fcps_status[:, -1] <= 0)[0])
        self.av_fcps_status[available_fc_index] = 0

        self.av_scps_status[:, -2] = current_t
        self.av_scps_status[:, -1] = self.av_scps_status[:, 1] - current_t
        available_sc_index = list(np.where(self.av_scps_status[:, -1] <= 0)[0])
        self.av_scps_status[available_sc_index] = 0

        self.av_ops += len(available_p_index)
        self.av_fcps += len(available_fc_index)
        self.av_scps += len(available_sc_index)

        self.supply_status = np.concatenate((self.av_ops_status, self.av_fcps_status, self.av_scps_status), axis=0)

    def update_cruising_t(self):
        # self.cruising_t = 15 * (1 - (self.av_fcps + self.av_ops + self.av_scps) / self.total_num)
        occ_rate = 1 - (self.av_ops + self.av_scps + self.av_fcps) / self.total_num
        self.cruising_t = min(30, 4.467 * np.power(occ_rate, 18.86))


class Env:
    def __init__(self, evaluate=False, park_arrival_num=100, charge_ratio=1, rule=1):
        self.action_space = 13
        self.step_supply_states = []
        self.accumulative_rewards = None
        self.plm = None
        self.evaluate = evaluate
        self.rule = rule
        self.park_arrival_num = park_arrival_num
        self.charge_ratio = charge_ratio

        self.observation_space = 100*6+4
        self.cruise_cost = None
        self.total_revenue = None
        self.t = 0
        self.ith_req = 0
        # self.t_next = 15
        self.states = np.zeros((self.observation_space, 1)).flatten()
        self.rewards = 0
        self.acc = 0
        self.r_p_no_avail = 0
        self.r_c_no_avail = 0

        self.termination = False  # 终止态

    def reset(self):
        self.req_info = get_train_req(self.park_arrival_num, self.charge_ratio)
        self.req_info = self.req_info.sort_values(by='arrival_t')  # sort by arrive time
        # 初始化plm
        self.plm = [ParkingLotManagement(i, self.req_info) for i in range(pl_num)]
        # 状态空间定义为供给和需求
        self.ith_req = 0
        self.t = self.req_info['arrival_t'].iloc[self.ith_req]
        self.req_id_at_t = self.req_info['req_id'].iloc[self.ith_req]

        self.supply_t = np.array([self.plm[i].init_ps_status for i in range(pl_num)])  # 4 * 100 * 6
        self.req_t = self.req_info[['arrival_t', 'leave_t', 'new_label', 'revenue_std']].iloc[self.ith_req].values
        self.states = self.req_t
        for j in range(len(self.supply_t)):
            self.states = np.concatenate((self.states,self.supply_t[j].flatten()))

        self.action_space = 13  # 动作空间维度 设计为13个 (1-3:1号停车场停车快充和慢充，4-6：2号停车场，7-9：3号停车场，10-12：4号停车场 0：拒绝时返回)

        self.rewards = 0  # 奖励函数，先设置为收益
        self.accumulative_rewards = 0
        self.park_revenue = 0  # 停车收益
        self.char_revenue = 0  # 充电收益
        self.total_revenue = 0
        self.travel_cost = 0  # 出行时间
        self.cruise_cost = 0  # 出行时间
        self.total_refuse = 0  # 拒绝数量
        self.park_refuse = 0  # 停车拒绝
        self.char_refuse = 0  # 充电拒绝
        self.step_supply_states = []  # 每个时间内的可用泊位数(step之后 把pl_current_supply添加进去即可) 可以获得占有率变化 这个只需要在评估的时候记录即可

        self.termination = False  # 终止态

        return self.states

    """
    有需求 但是拒绝分配时 设计动作为 0 reward设计为-5
    返回的mask值不一样
    """

    def get_invalid_action(self):
        mask = np.ones(self.action_space, dtype=int)
        mask_list = []

        for i in range(len(self.plm)):
            if self.req_t[2] == 0:  # 停车请求
                if len(list(np.where(self.plm[i].av_ops_status[:, -1] <= 0)[0])) == 0:  # ops 没空位
                    mask_list.append(i * 3 + 1)
                if len(list(np.where(self.plm[i].av_fcps_status[:, -1] <= 0)[0])) == 0:  # fcps 没空位
                    mask_list.append(i * 3 + 2)
                if len(list(np.where(self.plm[i].av_scps_status[:, -1] <= 0)[0])) == 0:  # scps 没空位
                    mask_list.append(i * 3 + 3)
            elif self.req_t[2] == 1:  # 充电请求
                mask_list.extend([1, 4, 7, 10, 3, 6, 9, 12])  # 先把停车资源和慢充资源排除
                if len(list(np.where(self.plm[i].av_fcps_status[:, -1] <= 0)[0])) == 0:  # fcps 没空位
                    mask_list.extend([i * 3 + 2])
            else:
                mask_list.extend([1, 4, 7, 10, 2, 5, 8, 11])  # 先把停车资源和快充资源排除
                if len(list(np.where(self.plm[i].av_scps_status[:, -1] <= 0)[0])) == 0:
                    mask_list.extend([i * 3 + 3])

        mask_list = list(set(mask_list)) # 去掉重复元素
        mask[mask_list] = 0
        return mask

    """
    需要重写 暂时不改
    """

    def reward_shaping_function(self, action):

        opt_action = self.optimal_policy['opt_action'].loc[self.req_id_at_t]
        std_rev = self.req_info['revenue_std'].loc[self.req_id_at_t]

        alpha = 1

        if action == 13:
            return 5
        else:
            if opt_action == 0:
                if action != opt_action:
                    return -4 + std_rev  # policy reward = -4
                else:
                    return 5 + std_rev  # policy reward = 5
            else:
                if action == 0:
                    return -4 - std_rev
                else:
                    if action == opt_action:
                        return 5 + std_rev
                    else:
                        return -4 + std_rev

    # 选一个停车场后更新相关信息
    def step(self, action):

        # 下一步开始前 对停车场状态进行更新
        for plmi in self.plm:
            plmi.update_supply_status(self.t)

        if self.ith_req < len(self.req_info)-1:

            if action != 0:
                # 添加请求
                self.plm[int((action - 1) // 3)].add_req(req_id=self.req_id_at_t, action_type=action)
                # 更新停车场状态和巡游时间
                for plmi in self.plm:
                    plmi.update_supply_status(self.t)
                    plmi.update_cruising_t()

                self.total_revenue += self.req_t[-1]
                self.cruise_cost += self.plm[int((action - 1) // 3)].cruising_t
                # self.travel_cost += cost_matrix[int((action - 1) // 3)][int(this_demand[1])] + 2 * \
                #                     cost_matrix[int((action - 1) // 3)][int(this_demand[2]) + 2]

                # 奖励函数设置
                self.rewards = self.acc
                # self.rewards = self.req_info['revenue_std'].loc[self.req_id_at_t]
                # self.rewards = self.reward_shaping_function(action)

                # 请求种类
                if self.req_t[2] == 0:
                    self.park_revenue += self.req_t[-1]
                else:
                    self.char_revenue += self.req_t[-1]

            else:  # 没有分配泊位

                for plmi in self.plm:
                    plmi.update_cruising_t()

                # self.rewards = self.reward_shaping_function(action)

                self.total_refuse += 1
                if self.req_t[2] == 0:
                    self.park_refuse += 1
                    self.rewards = self.r_p_no_avail
                    # self.rewards = -self.req_info['revenue_std'].loc[self.req_id_at_t]
                    # self.rewards = self.reward_shaping_function(action)
                else:
                    self.char_refuse += 1
                    self.rewards = self.r_c_no_avail
                    # self.rewards = -self.req_info['revenue_std'].loc[self.req_id_at_t]
                    # self.rewards = self.reward_shaping_function(action)

            # 更新到下一个请求
            self.ith_req += 1
            self.t = self.req_info['arrival_t'].iloc[self.ith_req]
            self.req_id_at_t = self.req_info['req_id'].iloc[self.ith_req]
            self.req_t = self.req_info[['arrival_t', 'leave_t', 'new_label', 'revenue_std']].iloc[self.ith_req].values
            # 更新状态
            self.supply_t = np.array([self.plm[i].supply_status for i in range(pl_num)])
            self.states = self.req_t
            for j in range(len(self.supply_t)):
                self.states = np.concatenate((self.states, self.supply_t[j].flatten()))

            self.step_supply_states.append(self.supply_t)
            self.accumulative_rewards += self.rewards

            return self.states, self.rewards, self.termination

        else:
            self.step_supply_states.append(self.supply_t)
            self.termination = True
            self.rewards = 0

            return self.states, self.rewards, self.termination
