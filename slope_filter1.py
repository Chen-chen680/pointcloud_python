# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:18
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : slope_filter1.py
# @Software: PyCharm


import numpy as np

def slope_filter(cloud, step, eldif_thre=0.5, slope_thre=0.1):
    """
    基于改进的数学形态学的坡道滤波算法，剔除树木、杂草等非地面点
    :param cloud: 输入的点云
    :param step: 格网边长
    :param eldif_thre:高差阈值
    :param slope_thre: 坡度阈值
    :return: 地面点索引
    """
    # --------------------创建格网-------------------------
    point_cloud = np.asarray(cloud.points)
    # 1、获取点云数据边界
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
    # 2、计算格网行列数
    width = np.ceil((x_max - x_min) / step)
    height = np.ceil((y_max - y_min) / step)
    # print("格网的大小为： {} x {}".format(width, height))
    # 3、计算每个点的格网索引
    h = list()  # h 为保存索引的列表
    for i in range(len(point_cloud)):

        col = np.ceil((point_cloud[i][0] - x_min) / step)
        row = np.ceil((point_cloud[i][1] - y_min) / step)
        h.append((row-1) * width + col)
    h = np.array(h)
    # 4、计算每个格网里点的高差、坡度
    h_indice = np.argsort(h)  # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    ground_idx = []  # 地面点索引
    begin = 0
    for i in range(len(h_sorted) - 1):
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            point_idx = h_indice[begin: i + 1]
            z_value = point_cloud[[point_idx], 2]
            z_min_idx = np.argmin(z_value)  # 获取格网内的最低点的z值

            delth = point_cloud[[point_idx], 2] - point_cloud[[point_idx[z_min_idx]], 2]  # 计算高差
            deltx = point_cloud[[point_idx], 0] - point_cloud[[point_idx[z_min_idx]], 0]
            delty = point_cloud[[point_idx], 1] - point_cloud[[point_idx[z_min_idx]], 1]
            distances = np.sqrt(deltx * deltx + delty * delty)
            # 计算坡度，分子分母为0的项即为最低点，直接赋值为0
            slope = np.divide(delth, distances, out=np.zeros_like(delth), where=distances != 0)
            # 坡度法实现过程
            for k in range(len(point_idx)):
                if (delth[0][k] < eldif_thre) & (slope[0][k] < slope_thre):
                    ground_idx.append(point_idx[k])
        begin = i + 1
    # 5、获取地面点索引
    ground_points = (cloud.select_by_index(ground_idx))
    other_points = (cloud.select_by_index(ground_idx, invert=True))
    return ground_points, other_points