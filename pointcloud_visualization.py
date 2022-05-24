# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:24
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : pointcloud_visualization.py
# @Software: PyCharm

import numpy as np
import open3d as o3d
import glob
import os

def las2pcd(las_cloud):
    '''将las格式的文件转化成pcd格式'''
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose() # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose() # 获取颜色
    pcd = o3d.geometry.PointCloud() # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz) # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color) # 给pcd对象添加颜色
    return pcd


def las_visualization(las_cloud):
    """las格式点云可视化"""
    pcd = las2pcd(las_cloud)
    pcd_visualization(pcd)


def pcd_visualization(pcd):
    """pcd格式点云可视化"""
    o3d.visualization.draw_geometries([pcd])

def bacth_pcd_viewer(path):
    '''批量pcd数据可视化，将多个pcd文件放在一起显示，用于结构面提取后'''
    path_list = glob.glob(os.path.join(path, '*.pcd'))
    for idx, pcd_path in enumerate(path_list):
        exec('pcd' + str(idx) + '= o3d.io.read_point_cloud(pcd_path)')
        exec('pcd' + str(idx) + '.paint_uniform_color(' + str([round(random.uniform(0,1), 1), round(random.uniform(0,1), 1), round(random.uniform(0,1), 1)]) +')')
    #------------------------------------创建绘图-------------------------------------------
    execution = 'o3d.visualization.draw_geometries(['
    for i in range(len(path_list)):
        execution = execution + 'pcd' + str(i) + ','
    execution = execution[0: -1] + '])'
    exec(execution)