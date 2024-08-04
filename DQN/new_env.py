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
fast_charge_num = [pl1.fast_charge_space, pl2.fast_charge_space, pl3.fast_charge_space, pl4.fast_charge_space, ]
slow_charge_num = [pl1.slow_charge_space, pl2.slow_charge_space, pl3.slow_charge_space, pl4.slow_charge_space, ]


def get_revenue(req_info):
    park_fee = pl1.park_fee / 2  # 半个小时的费用
    charge_fee = [0, pl1.fast_charge_fee, pl1.slow_charge_fee]  # 每分钟的价格
    reserved_fee = pl1.reserve_fee  # 预约费用
    req_info['revenue'] = [(np.floor(req_info['activity_t'].iloc[i] / 15) * park_fee + (
            req_info['activity_t'].iloc[i] * (charge_fee[req_info['new_label'].iloc[i]])) + reserved_fee) for i in
                           range(len(req_info))]
    return req_info


def get_train_req(park_arrival_num, charge_ratio):
    # 更新需求
    # req_info = demand.main(park_arrival_num, charge_ratio, train=True)
    req_info = pd.read_csv(r"G:\2023-纵向\停车分配\需求分布\demand0607\400-0.25.csv")
    return get_revenue(req_info)


def get_eval_req():
    evaluate_req_info = pd.read_csv(r"/需求分布/demand data/500-0.2.csv")
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
        self.park2charge_request_list = []
        self.cruising_t = 0
        self.req_info = req_information

    def add_req(self, req_id, add_type):  # 添加信息: 请求id 开始停车时间 活动时间 离开时间
        self.park_info.append(req_id)
        self.add_type.append(add_type)
        if add_type == 0:
            self.av_ops -= 1
        elif add_type == 1:
            self.av_fcps -= 1
        else:
            self.av_scps -= 1

    def available_supply_t(self):
        return [self.av_ops, self.av_fcps, self.av_scps]

    def remove_req(self, current_t):  # 移除停放到时间的请求
        try:
            for ith, req_id in enumerate(self.park_info):
                if self.req_info['leave_t'].loc[req_id] < current_t:
                    temp_label = self.add_type[ith]
                    self.park_info.remove(req_id)
                    self.add_type.remove(self.add_type[ith])
                    if temp_label == 0:
                        self.av_ops += 1
                    elif temp_label == 1:
                        self.av_fcps += 1
                    else:
                        self.av_scps += 1
        except:
            return 0

    def update_cruising_t(self):
        # self.cruising_t = 15 * (1 - (self.av_fcps + self.av_ops + self.av_scps) / self.total_num)
        occ_rate = 1 - (self.av_ops + self.av_scps + self.av_fcps) / self.total_num
        self.cruising_t = min(30, 4.467 * np.power(occ_rate, 18.86))


class Env:
    def __init__(self, evaluate=False, park_arrival_num=100, charge_ratio=1, rule=1):
        self.action_space = 14
        self.supply_t = []
        self.accumulative_rewards = None
        self.plm = None
        self.evaluate = evaluate
        self.rule = rule
        self.park_arrival_num = park_arrival_num
        self.charge_ratio = charge_ratio
        self._max_episode_steps = None
        self.observation_space = 18
        self.episode = 1440
        self.cruise_cost = None
        self.total_revenue = None
        self.t = 0
        self.states = 0  # 以上所有信息concatenate后作为状态
        self.rewards = 0  # 奖励函数，设置为出行时间，巡游时间之和，并用做标准化处理
        self.park_revenue = 0  # 停车收益
        self.char_revenue = 0  # 充电收益
        self.travel_cost = 0  # 出行时间
        self.total_refuse = 0  # 拒绝数量
        self.park_refuse = 0  # 停车拒绝
        self.char_refuse = 0  # 充电拒绝

        self.termination = 0  # 终止态
        self.done = 0  # 结束态

    def reset(self):
        self.req_info = get_train_req(self.park_arrival_num, self.charge_ratio)
        self.req_info = self.req_info.sort_values(by='request_t')
        # 初始化plm
        self.plm = [ParkingLotManagement(i, self.req_info) for i in range(pl_num)]
        # 状态空间定义为供给和需求
        self.plm_supply_t = np.array([self.plm[i].available_supply_t() for i in range(pl_num)]).reshape(pl_num,
                                                                                                        3)  # 3类资源

        self.total_demand_at_t = self.get_request_demand()  # t时间的总需求
        self.ith_demand_at_t = 0  # t时刻给出的第i个需求
        self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]  # 停车或充电需求  1*5

        self.states = np.concatenate(
            (np.array([self.t]).flatten(), self.plm_supply_t.flatten(), self.request_demand.flatten()))
        self.action_space = 14  # 动作空间维度 设计为14个 (1-3:1号停车场停车快充和慢充，4-6：2号停车场，7-9：3号停车场，10-12：4号停车场，13：无需求时返回 0：拒绝时返回)
        self._max_episode_steps = 1440
        self.t = 0  # 开始时刻为0时刻
        self.req_id_at_t = 0

        self.observation_space = len(self.states)

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
        self.supply_t = []  # 每个时间内的可用泊位数(step之后 把pl_current_supply添加进去即可) 可以获得占有率变化 这个只需要在评估的时候记录即可

        self.termination = False  # 终止态
        self.done = False  # 结束态

        return self.states

    # 停车或充电需求，包括需求种类，起终点编号（独热编码），活动时长
    def get_request_demand(self):
        result = self.req_info[['arrival_t', 'leave_t', 'O', 'D', 'new_label', 'revenue', 'req_id']].loc[
            self.req_info['arrival_t'] == self.t].values
        if len(result) == 0:
            self.req_id_at_t = 0
            return np.zeros((1, 5)).astype(int)  # 如果没有请求 则需求信息全部传回零
        else:
            self.req_id_at_t = result[0, 6]
            return result[:, 1:-1]  # 不把arrival_t 和 req id传过去


    """
    没有需求时 单独设计一个动作即为 13 reward设计为0
    有需求 但是拒绝分配时 设计动作为 0 reward设计为-5
    返回的mask值不一样
    """

    def get_invalid_action(self):
        mask = np.ones(self.action_space,dtype=int)
        demand = self.request_demand
        # 如果某个请求的收益为0 说明没有请求发送 此时为空的请求
        if demand[4] == 0:
            mask_list = list(range(0, 13))
            mask[mask_list] = 0
            return mask
        else:
            request_type = demand[3]
            supply = self.plm_supply_t
            add_mask = np.where(supply[:, int(request_type)] == 0)[0]
            if request_type == 0:
                mask_list = [2, 3, 5, 6, 8, 9, 11, 12, 13]
                if len(add_mask) > 0:
                    mask_list.extend([each * 3 + 1 for each in add_mask])
                mask[mask_list] = 0
                return mask
            elif request_type == 1:
                mask_list = [1, 3, 4, 6, 7, 9, 10, 12, 13]
                if len(add_mask) > 0:
                    mask_list.extend([each * 3 + 2 for each in add_mask])
                mask[mask_list] = 0
                return mask
            else:
                mask_list = [1, 3, 4, 5, 7, 8, 10, 11, 13]
                if len(add_mask) > 0:
                    mask_list.extend([each * 3 + 3 for each in add_mask])
                mask[mask_list] = 0
                return mask

    # 选一个停车场后更新相关信息
    def step(self, action):
        if self.t < self.episode:
            if 0 < action < 13:  # 有分配泊位
                this_demand = self.request_demand
                self.ith_demand_at_t += 1
                if self.rule == 1:
                    if action <= 3:
                        if action == 1:
                            self.plm[0].add_req(req_id=self.req_id_at_t, add_type=0)
                        elif action == 2:
                            self.plm[0].add_req(req_id=self.req_id_at_t, add_type=1)
                        elif action == 3:
                            self.plm[0].add_req(req_id=self.req_id_at_t, add_type=2)
                    elif action <= 6:
                        if action == 4:
                            self.plm[1].add_req(req_id=self.req_id_at_t, add_type=0)
                        elif action == 5:
                            self.plm[1].add_req(req_id=self.req_id_at_t, add_type=1)
                        elif action == 6:
                            self.plm[1].add_req(req_id=self.req_id_at_t, add_type=2)
                    elif action <= 9:
                        if action == 7:
                            self.plm[2].add_req(req_id=self.req_id_at_t, add_type=0)
                        elif action == 8:
                            self.plm[2].add_req(req_id=self.req_id_at_t, add_type=1)
                        elif action == 9:
                            self.plm[2].add_req(req_id=self.req_id_at_t, add_type=2)
                    else:
                        if action == 10:
                            self.plm[3].add_req(req_id=self.req_id_at_t, add_type=0)
                        elif action == 11:
                            self.plm[3].add_req(req_id=self.req_id_at_t, add_type=1)
                        elif action == 12:
                            self.plm[3].add_req(req_id=self.req_id_at_t, add_type=2)

                for plmi in self.plm:
                    plmi.remove_req(self.t)
                    plmi.update_cruising_t()

                req_type = this_demand[3]  # new label

                self.total_revenue += this_demand[4]
                self.cruise_cost += self.plm[int((action - 1) // 3)].cruising_t
                self.travel_cost += cost_matrix[int((action - 1) // 3)][int(this_demand[1])] + 2 * \
                                    cost_matrix[int((action - 1) // 3)][int(this_demand[2]) + 2]
                # 奖励为巡游时间
                # self.rewards = - self.plm[action].cruising_t - cost_matrix[action][demand[1]] - 2 * cost_matrix[action][
                #     demand[2] + 2]
                # 奖励为收益
                # self.rewards = this_demand[4]
                self.rewards = 5

                if self.ith_demand_at_t == len(self.total_demand_at_t):
                    self.t += 1
                    self.ith_demand_at_t = 0
                    self.total_demand_at_t = self.get_request_demand()

                self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]
                self.plm_supply_t = np.array([self.plm[i].available_supply_t() for i in range(pl_num)]).reshape(pl_num,
                                                                                                                3)
                self.states = np.concatenate(
                    (np.array([self.t]).flatten(), self.plm_supply_t.flatten(), self.request_demand.flatten()))

                if req_type == 0:
                    self.park_revenue += this_demand[4]
                else:
                    self.char_revenue += this_demand[4]

            else:  # 没有分配泊位  # 这里要考虑没有分配泊位是因为什么 是没有需求还是单纯拒绝
                this_demand = self.request_demand
                self.ith_demand_at_t += 1
                req_type = this_demand[3]  # new label
                for plmi in self.plm:
                    plmi.remove_req(self.t)
                    plmi.update_cruising_t()

                if action == 13:
                    if this_demand[4] == 0:  # 没有需求
                        self.rewards = 0
                else:  # 有需求但是没分配
                    # 奖励为巡游时间
                    # self.rewards = -10
                    # 奖励为收益 即拒绝惩罚
                    # self.rewards = -10
                    self.total_refuse += 1
                    if req_type == 0:
                        self.park_refuse += 1
                        self.rewards = -10
                    else:
                        self.char_refuse += 1
                        self.rewards = -50

                if self.ith_demand_at_t == len(self.total_demand_at_t):
                    self.t += 1
                    self.ith_demand_at_t = 0
                    self.total_demand_at_t = self.get_request_demand()

                self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]

                self.states = np.concatenate(
                    (np.array([self.t]).flatten(), self.plm_supply_t.flatten(), self.request_demand.flatten()))

            # print(f"时刻:{self.t}")
            # print(f"泊位状态:{self.current_supply()}")
            self.supply_t.append(self.plm_supply_t)
            self.accumulative_rewards += self.rewards

            return self.states, self.rewards, self.termination, [self.total_revenue, self.park_revenue,
                                                                 self.char_revenue, self.total_refuse,
                                                                 self.park_refuse, self.char_refuse,
                                                                 self.travel_cost, self.cruise_cost, self.plm_supply_t]

        else:
            self.supply_t.append(self.plm_supply_t)
            self.termination = True
            self.accumulative_rewards += self.rewards

            return self.states, self.rewards, self.termination, [self.total_revenue, self.park_revenue,
                                                                 self.char_revenue, self.total_refuse,
                                                                 self.park_refuse, self.char_refuse,
                                                                 self.travel_cost, self.cruise_cost, self.supply_t]

# def test():
#     train_req_info = get_train_req()
#     print("11")

# test()
