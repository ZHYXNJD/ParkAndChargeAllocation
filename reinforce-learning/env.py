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


def get_train_req():
    # 更新需求
    req_info = demand.main(park_arrival_num=500, charge_ratio=0.2)
    return get_revenue(req_info)


def get_eval_req():
    evaluate_req_info = pd.read_csv(r"G:\2023-纵向\停车分配\需求分布\demand data\500-0.2.csv")
    return get_revenue(evaluate_req_info)


class ParkingLotManagement:
    def __init__(self, pl_id, req_information):
        self.req_info = None
        self.id = pl_id
        self.park_info = []
        self.av_ops = ordinary_num[self.id]
        self.av_fcps = fast_charge_num[self.id]
        self.av_scps = slow_charge_num[self.id]
        self.total_num = self.av_ops + self.av_fcps + self.av_scps
        self.cruising_t = 0
        self.req_info = req_information

    def add_req(self, req_id):  # 添加信息: 请求id 开始停车时间 活动时间 离开时间
        self.park_info.append(req_id)
        temp_label = self.req_info['new_label'].loc[req_id]
        if temp_label == 0:
            self.av_ops -= 1
        elif temp_label == 1:
            self.av_fcps -= 1
        else:
            self.av_scps -= 1

    def remove_req(self, current_t):  # 移除停放到时间的请求
        for req_id in self.park_info:
            if self.req_info['leave_t'].loc[req_id] < current_t:
                temp_label = self.req_info['new_label'].loc[req_id]
                self.park_info.remove(req_id)
                if temp_label == 0:
                    self.av_ops += 1
                elif temp_label == 1:
                    self.av_fcps += 1
                else:
                    self.av_scps += 1

    def update_cruising_t(self):
        # self.cruising_t = 15 * (1 - (self.av_fcps + self.av_ops + self.av_scps) / self.total_num)
        occ_rate = 1 - (self.av_ops + self.av_scps + self.av_fcps) / self.total_num
        self.cruising_t = 4.467 * np.power(occ_rate, 18.86)


# plm = [ParkingLotManagement(i) for i in range(pl_num)]


class Env:
    def __init__(self, evaluate=False):
        self.supply_t = []
        self.accumulative_rewards = None
        self.plm = None
        self.evaluate = evaluate
        self._max_episode_steps = None
        self.action_space = pl_num + 1
        self.observation_space = 380
        self.episode = None
        self.cruise_cost = None
        self.total_revenue = None
        self.req_id_at_t = None
        self.demand_done_at_t = None
        self.all_demand_at_t = None
        self.ith_demand_at_t = None
        self.future_o_arrival_and_d = None
        self.t_next = None
        self.t = 0  # 当前时刻
        self.req_info = None
        self.pl_current_supply = 0  # 当前时刻空闲普通泊位和空闲充电泊位
        self.pl_future_supply = 0  # 未来一段时间内释放的可用泊位
        self.future_o_arrival = 0  # 未来一段时间内到达o的数量
        self.future_d_demand = 0  # 未来一段时间内d的需求量
        self.request_demand = None  # 停车或充电需求，包括需求种类，起终点编号（独热编码），活动时长
        self.states = 0  # 以上所有信息concatenate后作为状态

        self.action_space_num = 0  # 动作空间维度，这里是停车场的数量

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
        # 如果不是评估环境 每次reset后都需要刷新数据
        if not self.evaluate:
            self.req_info = get_train_req()
        else:
            self.req_info = get_eval_req()
        # 初始化plm
        self.plm = [ParkingLotManagement(i, self.req_info) for i in range(pl_num)]
        # 当前时刻
        self.episode = 1440
        self.t = 0
        self.t_next = 15
        self.req_id_at_t = 0
        # 当前时刻供给
        self.pl_current_supply = self.current_supply()  # 当前时刻空闲普通泊位和空闲充电泊位   4*4
        self.pl_future_supply = self.future_supply()  # 未来一段时间内释放的可用泊位  需要在分配完之后计算            4*45
        self.future_o_arrival = self.future_arrival_and_d()[0]  # 未来一段时间内到达o的数量  o*45
        self.future_d_demand = self.future_arrival_and_d()[1]  # 未来一段时间内d的需求量     d*45
        self.total_demand_at_t = self.get_request_demand()  # t时间的总需求
        self.ith_demand_at_t = 0  # t时刻给出的第i个需求
        self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]  # 停车或充电需求，包括需求种类，起终点编号（独热编码），活动时长  1*4
        self.states = np.concatenate((self.pl_current_supply.flatten(), self.pl_future_supply.flatten(),
                                      self.future_o_arrival.flatten(), self.future_d_demand.flatten(),
                                      self.request_demand.flatten()))  # 以上所有信息concatenate后作为状态

        self.observation_space = len(self.states)
        self.action_space = pl_num + 1  # 动作空间维度，这里是停车场的数量+拒绝维度
        self._max_episode_steps = 1440

        self.rewards = 0  # 奖励函数，设置为出行时间，巡游时间之和，并用做标准化处理
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

    def current_supply(self):
        supply = np.zeros((pl_num, 4)).astype(int)  # 1时间 + 3类资源
        for i in range(pl_num):
            for j in range(3):
                if j == 0:
                    supply[i][j + 1] = self.plm[i].av_ops
                elif j == 1:
                    supply[i][j + 1] = self.plm[i].av_fcps
                else:
                    supply[i][j + 1] = self.plm[i].av_scps
        supply[:, 0] = self.t
        return supply

    def future_supply(self):
        # 需要有现在的时间 和停放的车辆预计离开时间即可
        future_t = np.arange(self.t, self.t + self.t_next)
        pl_ops_future_supply = np.zeros((pl_num, self.t_next)).astype(int)  # 估计未来15min内
        pl_fcps_future_supply = np.zeros((pl_num, self.t_next)).astype(int)  # 估计未来15min内
        pl_scps_future_supply = np.zeros((pl_num, self.t_next)).astype(int)  # 估计未来15min内
        for i in range(pl_num):
            for each_req in self.plm[i].park_info:
                leave_t = self.req_info['leave_t'].loc[each_req]
                label = self.req_info['new_label'].loc[each_req]
                for j, t_ in enumerate(future_t):
                    if leave_t == t_:
                        if label == 0:
                            pl_ops_future_supply[i][j] += 1
                        elif label == 1:
                            pl_scps_future_supply[i][j] += 1
                        else:
                            pl_fcps_future_supply[i][j] += 1

        pl_future_supply = np.concatenate((pl_ops_future_supply, pl_fcps_future_supply, pl_scps_future_supply), axis=1)
        return pl_future_supply

    def future_arrival_and_d(self):
        future_park_arrival = np.zeros((O, self.t_next)).astype(int)  # 估计未来15min内
        future_fc_arrival = np.zeros((O, self.t_next)).astype(int)
        future_sc_arrival = np.zeros((O, self.t_next)).astype(int)

        future_d_park_demand = np.zeros((D, self.t_next)).astype(int)  # 估计未来15min内
        future_d_fc_demand = np.zeros((D, self.t_next)).astype(int)
        future_d_sc_demand = np.zeros((D, self.t_next)).astype(int)

        temp_index = self.req_info[
            (self.req_info['arrival_t'] >= self.t) & (self.req_info['arrival_t'] < self.t + self.t_next)].index.tolist()
        for each_arrival in temp_index:
            temp_o = self.req_info['O'].loc[each_arrival]
            temp_d = self.req_info['D'].loc[each_arrival]
            temp_t = self.req_info['arrival_t'].loc[each_arrival] - self.t
            temp_label = self.req_info['new_label'].loc[each_arrival]
            if temp_label == 0:
                future_park_arrival[temp_o][temp_t] += 1
                future_d_park_demand[temp_d][temp_t] += 1
            elif temp_label == 1:
                future_fc_arrival[temp_o][temp_t] += 1
                future_d_fc_demand[temp_d][temp_t] += 1
            else:
                future_sc_arrival[temp_o][temp_t] += 1
                future_d_sc_demand[temp_d][temp_t] += 1

        future_arrival = np.concatenate((future_park_arrival, future_fc_arrival, future_sc_arrival), axis=1)
        future_d_demand = np.concatenate((future_d_park_demand, future_d_fc_demand, future_d_sc_demand), axis=1)

        return [future_arrival, future_d_demand]

    # 停车或充电需求，包括需求种类，起终点编号（独热编码），活动时长
    def get_request_demand(self):
        result = self.req_info[['new_label', 'O', 'D', 'activity_t', 'req_id']].loc[
            self.req_info['arrival_t'] == self.t].values
        if len(result) == 0:
            self.req_id_at_t = 0  # 不能使用
            return np.zeros((1, 4)).astype(int)
        else:
            self.req_id_at_t = result[0, 4]
            return result[:, :4]  # 不把req id传过去

    # 对该时刻没有空闲泊位的停车场进行屏蔽
    def get_invalid_action(self):
        demand = self.request_demand
        # 如果没有请求 即元素全部为0
        if demand.any() == 0:
            return list(np.arange(0, 4))
        else:
            supply = self.pl_current_supply  # 4*4
            return np.where(supply[:, demand[0] + 1] == 0)[0]

    # 选一个停车场后更新相关信息
    def step(self, action):
        if self.t < self.episode:
            if action != pl_num:  # 有分配泊位
                demand = self.request_demand
                self.ith_demand_at_t += 1
                for plmi in self.plm:
                    if plmi.id == action:
                        plmi.add_req(req_id=self.req_id_at_t)
                    plmi.remove_req(self.t)
                    plmi.update_cruising_t()
                req_type = demand[0]  # new label
                self.pl_current_supply = self.current_supply()
                self.pl_future_supply = self.future_supply()
                self.future_o_arrival = self.future_arrival_and_d()[0]
                self.future_d_demand = self.future_arrival_and_d()[1]

                if self.ith_demand_at_t == len(self.total_demand_at_t):
                    self.t += 1
                    self.ith_demand_at_t = 0
                    self.total_demand_at_t = self.get_request_demand()

                self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]

                self.states = np.concatenate((self.pl_current_supply.flatten(), self.pl_future_supply.flatten(),
                                              self.future_o_arrival.flatten(), self.future_d_demand.flatten(),
                                              self.request_demand.flatten()))

                self.total_revenue += self.req_info['revenue'].loc[self.req_id_at_t]
                self.cruise_cost += self.plm[action].cruising_t
                self.travel_cost += cost_matrix[action][demand[1]] + 2 * cost_matrix[action][demand[2] + 2]
                self.rewards = - self.plm[action].cruising_t - cost_matrix[action][demand[1]] - 2 * cost_matrix[action][
                    demand[2] + 2]
                if req_type == 0:
                    self.park_revenue += self.req_info['revenue'].loc[self.req_id_at_t]
                else:
                    self.char_revenue += self.req_info['revenue'].loc[self.req_id_at_t]

            else:  # 没有分配泊位  # 这里要考虑没有分配泊位是因为什么 是没有需求还是没有剩余泊位
                demand = self.request_demand
                self.ith_demand_at_t += 1
                req_type = demand[0]  # new label
                for plmi in self.plm:
                    plmi.remove_req(self.t)
                    plmi.update_cruising_t()
                self.pl_current_supply = self.current_supply()
                self.pl_future_supply = self.future_supply()
                self.future_o_arrival = self.future_arrival_and_d()[0]
                self.future_d_demand = self.future_arrival_and_d()[1]

                if self.ith_demand_at_t == len(self.total_demand_at_t):
                    self.t += 1
                    self.ith_demand_at_t = 0
                    self.total_demand_at_t = self.get_request_demand()

                self.request_demand = self.total_demand_at_t[self.ith_demand_at_t]

                self.states = np.concatenate((self.pl_current_supply.flatten(), self.pl_future_supply.flatten(),
                                              self.future_o_arrival.flatten(), self.future_d_demand.flatten(),
                                              self.request_demand.flatten()))

                if demand.any() == 0:
                    self.rewards = 0
                else:
                    self.rewards = -10
                    self.total_refuse += 1
                    if req_type == 0:
                        self.park_refuse += 1
                    else:
                        self.char_refuse += 1

            # print(f"时刻:{self.t}")
            # print(f"泊位状态:{self.current_supply()}")
            self.supply_t.append(self.pl_current_supply)
            self.accumulative_rewards += self.rewards
            if self.evaluate:
                return self.states, self.rewards, self.termination, [self.total_revenue, self.park_revenue,
                                                                     self.char_revenue, self.total_refuse,
                                                                     self.park_refuse, self.char_refuse,
                                                                     self.travel_cost, self.cruise_cost, self.supply_t]
            else:
                return self.states, self.rewards, self.termination

        else:
            self.supply_t.append(self.pl_current_supply)
            self.termination = True
            self.accumulative_rewards += self.rewards
            if self.evaluate:
                return self.states, self.rewards, self.termination, [self.total_revenue, self.park_revenue,
                                                                     self.char_revenue, self.total_refuse,
                                                                     self.park_refuse, self.char_refuse,
                                                                     self.travel_cost, self.cruise_cost, self.supply_t]
            else:
                return self.states, self.rewards, self.termination


def test():
    train_req_info = get_train_req()
    print("11")


test()
