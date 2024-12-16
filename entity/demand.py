import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
from entity.basic_data.ratio import Arrival, SlowFastChargeRatio, ShortLongParkRatio
from entity import OD

matplotlib.use('TkAgg')


def truncated_gaussian(mean, std):
    a, b = (0 - mean) / std, np.inf
    return int(truncnorm(a, b, loc=mean, scale=std).rvs(size=1)[0])


# 这个函数用来计算大概的提交预约信息的时间
# 要根据到达时间和旅行时间推算
def request_time_dis(park_arrival_dis, charge_arrival_dis):
    request_for_park_dis = []
    request_for_charge_dis = []
    park_travel_time = []
    charge_travel_time = []
    # 出行时间和对应的概率
    travel_time = [10 / 60, 15 / 60, 25 / 60, 35 / 60, 45 / 60, 55 / 60, 70 / 60, 90 / 60, 120 / 60]  # 单位:h
    weight = [0.143, 0.288, 0.243, 0.12, 0.069, 0.068, 0.041, 0.018, 0.01]
    park_travel_time.extend(np.random.choice(travel_time, p=weight, size=len(park_arrival_dis)))
    charge_travel_time.extend(np.random.choice(travel_time, p=weight, size=len(charge_arrival_dis)))

    request_for_park_dis.extend([park_arrival_dis[i] - park_travel_time[i] for i in range(len(park_arrival_dis))])
    request_for_charge_dis.extend(
        [charge_arrival_dis[i] - charge_travel_time[i] for i in range(len(charge_arrival_dis))])
    return request_for_park_dis, request_for_charge_dis


# 当请求数量较少时 乘以到达比例后 会出现相加不等于初始化总数的情况
def sum_to_total(init_arrival, total_arrival):
    hourly_arrivals = np.round(init_arrival).astype(int)
    total_sum = np.sum(hourly_arrivals)
    # 如果总和小于总人数，增加最大值的值
    if total_sum < total_arrival:
        max_index = np.argmax(init_arrival)
        hourly_arrivals[max_index] += total_arrival - total_sum

    # 如果总和大于总人数，减少最大值的值
    elif total_sum > total_arrival:
        max_index = np.argmax(init_arrival)
        hourly_arrivals[max_index] -= total_sum - total_arrival

    return hourly_arrivals


class Demand:

    def __init__(self, park_arrival_num, charge_num):
        self.fast_std = 25
        self.fast_mean = 45
        self.slow_std = 280
        self.slow_mean = 100
        self.m_std = 85
        self.m_mean = 350
        self.l_std = 120
        self.l_mean = 650
        self.s_std = 60
        self.s_mean = 80
        self.el_std = 140
        self.el_mean = 1250
        self.park_num = park_arrival_num
        self.charge_num = charge_num

    def set_short_params(self, mean, std):
        self.s_mean = mean
        self.s_std = std

    def set_long_params(self, mean, std):
        self.l_mean = mean
        self.l_std = std

    def set_middle_params(self, mean, std):
        self.m_mean = mean
        self.m_std = std

    def set_slow_charge_params(self, mean, std):
        self.slow_mean = mean
        self.slow_std = std

    def set_fast_charge_params(self, mean, std):
        self.fast_mean = mean
        self.fast_std = std

    def arrival_per_hour(self):
        park_arrival_per_hour = []
        charge_arrival_per_hour = []
        park_arrival_ratio = Arrival().park_ratio
        charge_arrival_ratio = Arrival().charge_ratio

        for each in park_arrival_ratio:
            park_arrival_per_hour.append(self.park_num * each)
        for each in charge_arrival_ratio:
            charge_arrival_per_hour.append(self.charge_num * each)
        park_arrivals = sum_to_total(park_arrival_per_hour, self.park_num)
        charge_arrivals = sum_to_total(charge_arrival_per_hour, self.charge_num)

        return park_arrivals, charge_arrivals

    def park_arrival_dis(self):  # 每分钟的到达假设服从均匀分布  不用泊松分布是因为 初始值跟最终生成的值数量上不一样
        # 先得到每个小时到的数量
        park_arrival_per_hour, _ = self.arrival_per_hour()
        arrival_times_hours = []  # 存储小时单位的到达时间
        for i in range(len(park_arrival_per_hour)):
            # if i == 0:
            #     arrival_times_hours.extend(np.sort(np.random.uniform(0, 7, park_arrival_per_hour[0])))
            # else:
            #     arrival_times_hours.extend(np.sort(np.random.uniform(i + 6, i + 7, park_arrival_per_hour[i])))

            if i == 0:
                arrival_times_hours.extend(np.sort(np.random.uniform(0, 7, park_arrival_per_hour[i])))
            elif i == 1:
                arrival_times_hours.extend(np.sort(np.random.uniform(7, 8, park_arrival_per_hour[i])))
            elif i == 2:
                arrival_times_hours.extend(np.sort(np.random.uniform(8, 10, park_arrival_per_hour[i])))
            elif i == 3:
                arrival_times_hours.extend(np.sort(np.random.uniform(10, 12, park_arrival_per_hour[i])))
            elif i == 4:
                arrival_times_hours.extend(np.sort(np.random.uniform(12, 17, park_arrival_per_hour[i])))
            elif i == 5:
                arrival_times_hours.extend(np.sort(np.random.uniform(17, 20, park_arrival_per_hour[i])))
            elif i == 6:
                arrival_times_hours.extend(np.sort(np.random.uniform(20, 21, park_arrival_per_hour[i])))
            else:
                arrival_times_hours.extend(np.sort(np.random.uniform(21, 24, park_arrival_per_hour[i])))

        return arrival_times_hours

    def get_park_time_by_type(self, park_type):
        if park_type == 's':
            a = truncated_gaussian(self.s_mean, self.s_std)
            return a
        elif park_type == 'm':
            a = truncated_gaussian(self.m_mean, self.m_std)
            return a
        elif park_type == 'l':
            a = truncated_gaussian(self.l_mean, self.l_std)
            return a
        else:
            a = truncated_gaussian(self.el_mean, self.el_std)
            return a

    def park_duration_dis(self, arrival_times_hours):  # 返回停车时长的分布
        park_duration = []
        choice_ratio = ShortLongParkRatio().choice_ratio
        for each in arrival_times_hours:
            if each <= 4:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[0]))
            elif each <= 6:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[1]))
            elif each <= 8:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[2]))
            elif each == 9:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[3]))
            elif each == 10:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[4]))
            elif each <= 13:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[5]))
            elif each <= 16:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[6]))
            elif each <= 19:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[7]))
            else:
                pt = self.get_park_time_by_type(np.random.choice(['s', 'm', 'l', 'el'], p=choice_ratio[8]))
            park_duration.append(pt)

        return park_duration

    def charge_arrival_dis(self):  # 每分钟的到达假设服从均匀分布
        # 先得到每个小时到的数量
        _, charge_arrival_per_hour = self.arrival_per_hour()
        arrival_times_hours = []  # 存储小时单位的到达时间
        for i in range(len(charge_arrival_per_hour)):
            # arrival_times_hours.extend(np.sort(np.random.uniform(i, i + 1, charge_arrival_per_hour[i])))
            if i == 0:
                arrival_times_hours.extend(np.sort(np.random.uniform(0, 1, charge_arrival_per_hour[i])))
            elif i == 1:
                arrival_times_hours.extend(np.sort(np.random.uniform(1, 10, charge_arrival_per_hour[i])))
            elif i == 2:
                arrival_times_hours.extend(np.sort(np.random.uniform(10, 11, charge_arrival_per_hour[i])))
            elif i == 3:
                arrival_times_hours.extend(np.sort(np.random.uniform(11, 12, charge_arrival_per_hour[i])))
            elif i == 4:
                arrival_times_hours.extend(np.sort(np.random.uniform(12, 13, charge_arrival_per_hour[i])))
            elif i == 5:
                arrival_times_hours.extend(np.sort(np.random.uniform(13, 19, charge_arrival_per_hour[i])))
            elif i == 6:
                arrival_times_hours.extend(np.sort(np.random.uniform(19, 22, charge_arrival_per_hour[i])))
            else:
                arrival_times_hours.extend(np.sort(np.random.uniform(22, 24, charge_arrival_per_hour[i])))

        return arrival_times_hours

    def get_charge_time_by_type(self, charge_type):
        if charge_type == 'slow':
            return truncated_gaussian(self.slow_mean, self.slow_std)
        else:
            return truncated_gaussian(self.fast_mean, self.fast_std)

    def charge_duration_dis(self, arrival_times_hours):
        global ct
        # 5*2 的array
        choice_ratio = SlowFastChargeRatio().choice_ratio
        charge_duration = []
        for each in arrival_times_hours:
            if each <= 6:
                ct = self.get_charge_time_by_type(np.random.choice(['slow', 'fast'], p=choice_ratio[0]))
            elif each <= 9:
                ct = self.get_charge_time_by_type(np.random.choice(['slow', 'fast'], p=choice_ratio[1]))
            else:
                ct = self.get_charge_time_by_type(np.random.choice(['slow', 'fast'], p=choice_ratio[2]))
            charge_duration.append(ct)
        return charge_duration


def test_plot(p_arr, c_arr, p_t, c_t):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 绘制直方图和核密度图（使用Seaborn）
    sns.histplot(p_arr, bins=30, kde=True, ax=axes[0, 0], color='green', edgecolor='black')
    axes[0, 0].set_title('p_arr')

    sns.histplot(c_arr, bins=30, kde=True, ax=axes[0, 1], color='green', edgecolor='black')
    axes[0, 1].set_title('c_arr')

    sns.histplot(p_t, bins=30, kde=True, ax=axes[1, 0], color='green', edgecolor='black')
    axes[1, 0].set_title('p_t')

    sns.histplot(c_t, bins=30, kde=True, ax=axes[1, 1], color='green', edgecolor='black')
    axes[1, 1].set_title('c_t')

    # 调整布局
    plt.tight_layout()
    # 显示图形
    plt.show()


# 新增new_label列
def assign_new_label(row):
    if row['label'] == 0:
        return 0
    elif row['label'] == 1 and row['activity_t'] <= 60:
        return 1
    elif row['label'] == 1 and row['activity_t'] > 60:
        return 2


def all_info(park_num, charge_num, O, D, park_info, charge_info):
    # park_info: request time, arrival time, park time
    # charge_info: request time, arrival time,charge time

    all_ = pd.DataFrame(
        columns=['request_t', 'arrival_t', 'activity_t', 'leave_t', 'label', 'O', 'D', 'new_label', 'revenue',
                 'req_id'])
    label = [0] * park_num
    charge_label = [1] * int(charge_num)
    label.extend(charge_label)
    park_info[0].extend(charge_info[0])
    park_info[1].extend(charge_info[1])
    park_info[2].extend(charge_info[2])
    all_.iloc[:, 0] = park_info[0]  # request t
    all_.iloc[:, 1] = park_info[1]  # arrival t
    all_.iloc[:, 2] = park_info[2]  # activity t
    all_.iloc[:, 4] = label  # label
    all_.iloc[:, 5] = np.random.choice(range(O), size=len(label), p=[0.5, 0.5])  # o
    all_.iloc[:, 6] = np.random.choice(range(D), size=len(label), p=[0.5, 0.5])  # d
    all_['new_label'] = all_.apply(assign_new_label, axis=1)
    # 将到达时间转换为min
    all_['request_t'] = (all_['request_t'] * 60).astype(int)
    all_['arrival_t'] = (all_['arrival_t'] * 60).astype(int)
    all_.iloc[:, 3] = all_.iloc[:, 1] + all_.iloc[:, 2]
    all_['leave_t'] = all_['leave_t'].astype(int)
    all_['req_id'] = all_.index.tolist()
    return all_


def main(park_arrival_num, charge_number, plot=False, train=False):
    if not train:
        # 确保可以复现
        # 评估的时候打开随机种子  训练的时候关闭掉
        # np.random.seed(1000)
        np.random.seed(43)
        # np.random.seed(521)

    origin_num = OD.OdCost().O
    destination_num = OD.OdCost().D
    demand = Demand(park_arrival_num, charge_number)
    p_arr = demand.park_arrival_dis()
    c_arr = demand.charge_arrival_dis()
    p_t = demand.park_duration_dis(p_arr)
    c_t = demand.charge_duration_dis(c_arr)
    p_r, c_r = request_time_dis(p_arr, c_arr)

    if plot:
        test_plot(p_arr, c_arr, p_t, c_t)
        return 0
    # 训练的时候关闭
    if not train:
        demand_data = all_info(park_arrival_num, charge_number, origin_num, destination_num, [p_r, p_arr, p_t],
                               [c_r, c_arr, c_t])

        # demand_data.to_csv(fr"G:\2023-纵向\停车分配\需求分布\demand1125\{park_arrival_num}-{charge_ratio}.csv")
        demand_data.to_csv(fr"G:\2023-纵向\停车分配\需求分布\demand1125\{park_arrival_num}-{charge_number}-0.csv")
    # 训练的时候打开
    else:
        return all_info(park_arrival_num, charge_number, origin_num, destination_num, [p_r, p_arr, p_t],
                        [c_r, c_arr, c_t])


if __name__ == '__main__':
    # for each in [120,240,360,480]:
    #     main(park_arrival_num=each, charge_number=600-each)
    main(park_arrival_num=300, charge_number=300)
    # main(park_arrival_num=500,charge_ratio=1,plot=False)
