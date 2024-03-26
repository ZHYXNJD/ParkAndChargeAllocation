# fcfs策略的仿真程序
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm


matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 根据到达时间分布生成单位到达车辆数
# 需要混合高斯的参数
def arrival_per_hour(total_arrivals):
    # 生成到达分布的参数
    mu1, sigma1, weight1 = 8.337429, 0.8699825, 0.3785887
    mu2, sigma2, weight2 = 13.924265, 3.8023894, 0.6214113

    # 时间间隔，假设每小时一个数据点
    time_intervals = np.arange(0, 24, 1)

    # 计算混合高斯分布在每个时间间隔的值
    gaussian1 = norm.pdf(time_intervals, mu1, sigma1)
    gaussian2 = norm.pdf(time_intervals, mu2, sigma2)

    # 计算混合高斯函数的值
    mixture_gaussian = weight1 * gaussian1 + weight2 * gaussian2

    # 将混合高斯函数的值乘以总到达车辆数，得到每分钟的到达车辆数
    arrivals_per_hour = total_arrivals * mixture_gaussian / np.sum(mixture_gaussian)

    # 可以通过舍入或其他处理确保结果是整数
    arrivals_per_hour = np.round(arrivals_per_hour).astype(int)

    # 可以将结果可视化
    plt.plot(time_intervals, arrivals_per_hour)
    plt.xlabel('时间（小时）')
    plt.ylabel('到达车辆数')
    plt.title('模拟到达车辆数分布')
    plt.show()

    return arrivals_per_hour

# 生成停车时长的混合高斯分布
def generate_park_duration(arrival_per_hour):
    """
    根据指定的均值、标准差和权重，生成一维混合高斯分布的样本。
    """

    # 生成停车时长的示例参数
    weights = [0.15, 0.16, 0.06, 0.28, 0.35]  # 5个高斯分布的均值
    means = [7.222, 34.416, 543.988, 149.033, 490.832]  # 两个高斯分布的标准差
    stds = [5.661, 18.294, 31.945, 69.962, 194.372]  # 混合权重

    n_components = len(means)
    samples = []

    for _ in range(arrival_per_hour):
        # 根据权重选择一个高斯分布
        component = np.random.choice(n_components, p=weights)
        # 从选定的高斯分布中生成样本
        sample = np.random.normal(means[component], stds[component])
        # 如果生成的样本值小于0 则舍弃该值
        if sample > 0:
            samples.append(sample)

    # plt.hist(samples, bins=30, density=True, alpha=0.6)
    # plt.title('Histogram of Generated Samples from 1D GMM')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.show()

    return np.array(samples)


# 生成每个小时到的车辆停车分布
def park_info_per_hour(total_arrivals):
    park_info = []  # 行表示每小时，列表示停车时间
    arrivals_per_hour = arrival_per_hour(total_arrivals)   # 首先得到每个小时的到达车辆数
    for each in arrivals_per_hour:
        if each > 0:
            park_info.append(generate_park_duration(each))
        else:
            park_info.append(None)   # 如果该时段没有车辆进行停车 则将停车时间记为0

    print(park_info)



# 生成样本
# n_samples = 1000
# samples = generate_park_duration(1800)
# 总到达车辆数
total_arrivals = 1800
# arrivals_per_hour = arrival_per_hour(total_arrivals)
park_info_per_hour(total_arrivals)


