# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/24 23:27
# @Author  : ChenLei
# @Email   : 1251742511@qq.com
# @File    : euclideanseg_xyz.py
# @Software: PyCharm

import os
from utils.utils import  create_dir_w, after_seg_connect_small_clusters
from pclpy import pcl
import open3d as o3d

def EuclideanSeg_xyz(pcd_path, cluster_tolerance=0.5, min_cluster_size=10, fore_clusters_num=10, total_cluters_num=20):
    '''欧式距离分割法提取岩石结构面'''
    # -----------------------创建保存的文件夹----------------------------
    save_dir = os.path.join(os.path.dirname(pcd_path), 'euclidean_xyz')
    create_dir_w(save_dir)
    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZ()
    pcl.io.loadPCDFile(pcd_path, pc)
    # ------------------------------欧式聚类----------------------------------
    ec = pcl.segmentation.EuclideanClusterExtraction.PointXYZ()
    tree = pcl.search.KdTree.PointXYZ()
    ec.setInputCloud(pc)                      # 输入点云
    ec.setSearchMethod(tree)                  # 设置搜索方式为kdtree
    ec.setClusterTolerance(cluster_tolerance)                 # 设置近邻搜索的搜索半径为1m
    ec.setMaxClusterSize(100000000)               # 一个聚类需要的最大点数
    ec.setMinClusterSize(min_cluster_size)                  # 一个聚类需要的最小点数
    clusters = pcl.vectors.PointIndices()     # 构建保存分割结果的索引向量
    ec.extract(clusters)                      # 获取分割索引
    # ----------------------------输出相关信息---------------------------------
    max_cluster_count = max([len(c.indices) for c in clusters])
    print('聚类的个数为：', len(clusters))
    print('聚类结果中的最大点数为：', max_cluster_count)
    # -----------------------保存聚类结果到本地文件夹---------------------------
    extract = pcl.filters.ExtractIndices.PointXYZ()
    it = 0
    result_list = []
    for c in clusters:
        cloud_inliers = pcl.PointCloud.PointXYZ()
        inliers = pcl.PointIndices()
        # 构建分类点的索引
        for i in c.indices:
            inliers.indices.append(i)
        # 根据分类索引提取分类点云
        extract.setInputCloud(pc)
        extract.setIndices(inliers)
        extract.setNegative(False)  # 设置为false表示提取对应索引之内的点
        extract.filter(cloud_inliers)
        # 批量保存分类结果
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_inliers.xyz)
        result_list.append(pcd)
        it += 1
    after_seg_connect_small_clusters(result_list, total_cluters_num, save_dir, 'EuclideanSeg')