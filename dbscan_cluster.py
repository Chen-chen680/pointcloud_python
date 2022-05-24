# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:17
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : dbscan_cluster.py
# @Software: PyCharm

import os
from utils.utils import create_dir_w, after_seg_connect_small_clusters
import open3d as o3d
import numpy as np

def dbscan_cluster(pcd_path, eps=0.5, min_points=10, fore_clusters_num=10, total_clusters_num=20):
    '''密度聚类方法实现岩石结构面图区'''
    # ------------------- 创建保存的文件夹----------------------------
    save_path = os.path.join(os.path.dirname(pcd_path), 'dbscan')
    create_dir_w(save_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    # 设置为debug调试模式
        # -------------------密度聚类--------------------------
    labels = np.array(pcd.cluster_dbscan(eps=eps,               # 邻域距离
                                         min_points=min_points,          # 最小点数
                                         print_progress=False))  # 是否在控制台中可视化进度条
    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # ---------------------保存聚类结果------------------------
    result_list = []
    for i in range(max_label + 1):
        ind = np.where(labels == i)[0]
        clusters_cloud = pcd.select_by_index(ind)
        result_list.append(clusters_cloud)
    after_seg_connect_small_clusters(result_list, total_clusters_num=total_clusters_num, save_dir=save_path, method='dbscan')
