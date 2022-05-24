# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:21
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : plane_fitting.py
# @Software: PyCharm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import open3d as o3d
import glob
from utils.utils import get_min_max_3d
import numpy as np

def plane_fit_1():
    '''使用RANSAC拟合提取的结构面，并绘制出其结构面'''
    base_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data'
    huapo = 'tang'
    method = 'region_xyz'
    fig = plt.figure(figsize=(10, 8))
    pcd_basepath = os.path.join(base_path, huapo, method)
    pcd_path_list = glob.glob(os.path.join(pcd_basepath, '*.pcd'))
    for idx,pcd_path in enumerate(pcd_path_list):
        pcd = o3d.io.read_point_cloud(pcd_path)
        plane_model, inliers = pcd.segment_plane(distance_threshold=1,
                                                 ransac_n=30,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        pts_min, pts_max = get_min_max_3d(pcd)
        x_min, y_min, z_min = pts_min[0], pts_min[1], pts_min[2]
        x_max, y_max, z_max = pts_max[0], pts_max[1], pts_max[2]

        x = np.linspace(x_min, x_max, 1000)
        y = np.linspace(y_min, y_max, 1000)
        X,Y = np.meshgrid(x, y)
        print((f"Z = ((-{a:.2f}*(X) - {b:.2f}*(Y) - {d:.2f}) / {c:.2f})"))
        exec(f"Z = ((-{a:.2f}*(X) - {b:.2f}*(Y) - {d:.2f}) / {c:.2f})")
        exec("ax" + str(idx) + "= fig.gca(projection='3d')")
        exec("ax"+ str(idx) + ".plot_surface(X, Y, Z)")
    fig.show()