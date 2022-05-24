# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:19
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : statistics_remove_points.py
# @Software: PyCharm

import open3d as o3d

def statistics_remove_points(pcd, nb_neighbors=50, std_ratio=3):
    '''
    统计滤波算法
    :param pcd: pcd数据格式
    :param nb_neighbors: 多少个临近点
    :param std_ratio: 多少个std
    :return: 返回内部点、离群点
    '''
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    sor_cloud = pcd.select_by_index(ind)
    out_cloud = pcd.select_by_index(ind, invert=True)
    return sor_cloud, out_cloud