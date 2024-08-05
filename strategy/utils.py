"""
分配时用到的一些工具
"""
import json
import os
from datetime import datetime

import numpy as np

from entity import parkinglot

pl1, pl2, pl3, pl4 = parkinglot.get_parking_lot(parking_lot_num=4)
pl = [pl1, pl2, pl3, pl4]


def get_rmk(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, max(req_info['leave_t'])))
    for i in range(req_num):
        start = req_info['arrival_t'].iloc[i]
        end = req_info['leave_t'].iloc[i] + 1
        rmk[i, start:end] = 1
    return rmk


def get_rmk_(req_info):
    req_num = len(req_info)
    rmk = np.zeros((req_num, max(req_info['leave_t'])))
    for i in range(req_num):
        start = req_info['arrival_t'].loc[i]
        end = req_info['leave_t'].loc[i] + 1
        rmk[i, start:end] = 1
    return rmk


# 全部泊位分布
def T_ZN(Z, N, pl):
    t_zn = np.zeros((Z, N)).astype(dtype=np.int8)
    all_index = []
    each_index = []
    index = [0]
    temp_index = 0
    for i in range(Z):
        temp_index += pl[i].total_num
        index.append(temp_index)
        t_zn[i, index[i]:index[i + 1]] = 1
        all_index.extend(range(index[i], index[i + 1]))
        each_index.append(list(range(index[i], index[i + 1])))
    return t_zn, all_index, each_index


# 普通泊位分布
def P_ZN(Z, N, pl):
    p_zn = np.zeros((Z, N)).astype(dtype=np.int8)
    ops_index = []
    index = [0]
    temp_index = 0
    for i in range(Z):
        temp_index += pl[i].ordinary_num
        index.append(temp_index)
        p_zn[i, index[i * 2]:index[i * 2 + 1]] = 1
        ops_index.extend(range(index[i * 2], index[i * 2 + 1]))
        temp_index += pl[i].charge_num
        index.append(temp_index)
    return p_zn, ops_index


# 充电泊位分布
def C_ZN(Z, N, pl):
    c_zn = np.zeros((Z, N)).astype(dtype=np.int8)
    cps_index = []
    index = []
    temp_index = 0
    for i in range(Z):
        temp_index += pl[i].ordinary_num
        index.append(temp_index)
        temp_index += pl[i].charge_num
        index.append(temp_index)
        c_zn[i, index[(i * 2)]:index[i * 2 + 1]] = 1
        cps_index.extend(range(index[(i * 2)], index[i * 2 + 1]))

    return c_zn, cps_index


# 快充泊位分布
def Fast_ZN(Z, N, pl):
    fast_zn = np.zeros((Z, N)).astype(dtype=np.int8)
    index = []
    fast_index = []
    temp_index = 0
    for i in range(Z):
        temp_index += pl[i].ordinary_num
        index.append(temp_index)
        temp_index += pl[i].fast_charge_space
        index.append(temp_index)
        fast_zn[i, index[i * 2]:index[i * 2 + 1]] = 1
        fast_index.extend(range(index[i * 2], index[i * 2 + 1]))
        temp_index += pl[i].slow_charge_space
    return fast_zn, fast_index


# 慢充泊位分布
def Slow_ZN(Z, N, pl):
    slow_zn = np.zeros((Z, N)).astype(dtype=np.int8)
    index = []
    slow_index = []
    temp_index = 0
    for i in range(Z):
        temp_index += pl[i].ordinary_num + pl[i].fast_charge_space
        index.append(temp_index)
        temp_index += pl[i].slow_charge_space
        index.append(temp_index)
        slow_zn[i, index[i * 2]:index[i * 2 + 1]] = 1
        slow_index.extend(range(index[i * 2], index[i * 2 + 1]))
    return slow_zn, slow_index


def cruise_t(z, snk, index, arrival_t):
    occ_rate = np.sum(snk[index, arrival_t], axis=0) / pl[z].total_num
    return 4.467 * np.power(occ_rate, 18.86)


def save_episode_results(data_frame,supply,args):

    folder_name = str(datetime.now().strftime("%m-%d-%H-%M"))
    save_path = '../save_data_reinforce/' + folder_name
    os.makedirs(save_path)
    # 打开文件进行写入
    config = json.dumps(vars(args))
    with open(save_path+'/configs.txt','w') as f:
        f.write(config)
        f.write('\n')
    data_frame.to_csv(save_path + '/episode_results.csv')
    np.save(save_path + '/supply_pl.npy', supply)


def test():
    pl1 = parkinglot.Parkinglot(id=1, total_num=40, charge_num=10, slow_charge_num=3)
    pl2 = parkinglot.Parkinglot(id=2, total_num=20, charge_num=5, slow_charge_num=1)
    pl3 = parkinglot.Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)

    pzn, ops_index = C_ZN(3, 70, [pl1, pl2, pl3])


test()
