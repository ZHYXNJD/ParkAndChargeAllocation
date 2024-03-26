"""
这部分生成时间矩阵 即 O-parkinglot-D之间的距离
 --- 确定O的个数 D 的个数
 --- 确定点之间的距离
"""
import numpy as np

# 两个O 两个D 单位:min
"""
      O1   O2   D1   D2
pl1   3     5    1   5

pl2   3     5    3   3

pl3   5     3    10  1

pl4   5     3    5   10
"""


class OdCost:
    def __init__(self):
        self.O = 2
        self.D = 2
        self.pl = 4
        self.cost_matrix = np.array([[3, 5, 1, 5],
                                     [3, 5, 3, 3],
                                     [5, 3, 10, 1],
                                     [5, 3, 5, 10]])

    def get_od_info(self):
        return self.O, self.D, self.pl, self.cost_matrix
