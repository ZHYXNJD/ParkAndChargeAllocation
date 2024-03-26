import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def generate_mixed_gaussian_samples(n, means, stds, weights):
    """
    根据指定的均值、标准差和权重，生成一维混合高斯分布的样本。
    """
    n_components = len(means)
    samples = []

    for _ in range(n):
        # 根据权重选择一个高斯分布
        component = np.random.choice(n_components, p=weights)
        # 从选定的高斯分布中生成样本
        sample = np.random.normal(means[component], stds[component])
        samples.append(sample)

    return np.array(samples)


# 示例参数
means = [0, 5]  # 两个高斯分布的均值
stds = [1, 1.5]  # 两个高斯分布的标准差
weights = [0.5, 0.5]  # 混合权重

# 生成样本
n_samples = 1000
samples = generate_mixed_gaussian_samples(n_samples, means, stds, weights)

# 可视化
plt.hist(samples, bins=30, density=True, alpha=0.6)
plt.title('Histogram of Generated Samples from 1D GMM')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

