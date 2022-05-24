from pclpy import pcl
import math
import laspy
import open3d as o3d
import numpy as np
import os

# init
las_path = r'.\data\eight\cloud_18148ef5.las'  '''注意，pcl不能读取路径中存在中文的文件，吐血！！'''
out_pcd = r'.\data\eight\cloud_18148ef5.pcd'

def main():
    if os.path.exists(out_pcd) == False:
        pcd = get_pcd(las_path)  # 读取las数据，得到降采样之后的pcd数据
        o3d.io.write_point_cloud(out_pcd, pcd)
        print(1)
    reign_grow(out_pcd)


def my_uniform_down_sample(pcd):
    print("原始点云中点的个数为：", np.asarray(pcd.points).shape[0])
    # o3d.visualization.draw_geometries([pcd])
    print("每5个点来降采样一个点")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=100)
    print("下采样之后点的个数为：", np.asarray(uni_down_pcd.points).shape[0])
    return uni_down_pcd

def get_pcd(path):
    '''las数据可视化，并降采样'''
    las_cloud = laspy.read(path)
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose()  # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose()  # 获取颜色

    # 创建pcd对象
    pcd = o3d.geometry.PointCloud()  # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz)  # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color)  # 给pcd对象添加颜色

    down_pcd = my_uniform_down_sample(pcd) # 降采样
    return down_pcd # 返回降采样之后的点云数据

def reign_grow(path):
    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile(path, pc)

    # -------------------------法向量和表面曲率估计----------------------------
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZRGBA_Normal()
    normals_estimation.setInputCloud(pc)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(0.35)
    normals_estimation.compute(normals)

    # ------------------------------区域生长----------------------------------
    rg = pcl.segmentation.RegionGrowing.PointXYZRGBA_Normal()   # 创建区域生长分割对象
    rg.setInputCloud(pc)                                        # 输入点云
    rg.setInputNormals(normals)                                 # 输入法向量
    rg.setMaxClusterSize(500000)                               # 一个聚类需要的最大点数
    rg.setMinClusterSize(60)                                    # 一个聚类需要的最小点数
    rg.setNumberOfNeighbours(15)                                # 搜索的邻域点的个数
    rg.setSmoothnessThreshold(5 / 180 * math.pi)                # 设置平滑阈值，法线差值阈值
    rg.setCurvatureThreshold(5)                                 # 设置表面曲率的阈值
    rg.setResidualThreshold(1)                                  # 设置残差阈值
    clusters = pcl.vectors.PointIndices()                       # 构建保存分割结果的索引向量
    rg.extract(clusters)                                        # 获取分割索引
    # ----------------------------输出相关信息---------------------------------
    max_cluster_count = max([len(c.indices) for c in clusters])
    print('聚类的个数为：', len(clusters))
    print('聚类结果中的最大点数为：', max_cluster_count)
    # -----------------------保存聚类结果到本地文件夹---------------------------
    extract = pcl.filters.ExtractIndices.PointXYZRGBA()
    it = 0

    for c in clusters:
        cloud_inliers = pcl.PointCloud.PointXYZRGBA()
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
        file_name = "RegionGrong_"+str(it)+".pcd"
        # pcl.io.savePCDFileBinary(file_name, cloud_inliers)
        pcl.io.savePCDFileASCII(os.path.join('./results', file_name), cloud_inliers)
        it += 1

    # ---------------------可视化聚类结果----------------------------
    # colored_cloud = pcl.PointCloud.PointXYZRGBA()
    # colored_cloud = rg.getColoredCloud()
    # viewer = pcl.visualization.PCLVisualizer("RegionGrowing")
    # viewer.setBackgroundColor(0, 0, 0)
    # viewer.addPointCloud(colored_cloud, "RegionGrowing cloud")
    # viewer.setPointCloudRenderingProperties(0, 1, "RegionGrowing cloud")
    # while not viewer.wasStopped():
    #     viewer.spinOnce(10)


def EuclideanSeg():
    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZRGB()
    pcl.io.loadPCDFile(out_pcd, pc)
    # ------------------------------欧式聚类----------------------------------
    ec = pcl.segmentation.EuclideanClusterExtraction.PointXYZRGB()
    tree = pcl.search.KdTree.PointXYZRGB()
    ec.setInputCloud(pc)                      # 输入点云
    ec.setSearchMethod(tree)                  # 设置搜索方式为kdtree
    ec.setClusterTolerance(0.5)                 # 设置近邻搜索的搜索半径为1m
    ec.setMaxClusterSize(1000000000)               # 一个聚类需要的最大点数
    ec.setMinClusterSize(50)                  # 一个聚类需要的最小点数
    clusters = pcl.vectors.PointIndices()     # 构建保存分割结果的索引向量
    ec.extract(clusters)                      # 获取分割索引
    # ----------------------------输出相关信息---------------------------------
    max_cluster_count = max([len(c.indices) for c in clusters])
    print('聚类的个数为：', len(clusters))
    print('聚类结果中的最大点数为：', max_cluster_count)
    # -----------------------保存聚类结果到本地文件夹---------------------------
    extract = pcl.filters.ExtractIndices.PointXYZRGB()
    it = 0
    for c in clusters:
        cloud_inliers = pcl.PointCloud.PointXYZRGB()
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
        file_name = "./eucli/EuclideanCluster_"+str(it)+".pcd"
        # pcl.io.savePCDFile(file_name, cloud_inliers)
        pcl.io.savePCDFileASCII(file_name, cloud_inliers)
        it += 1

if __name__ == '__main__':
    # EuclideanSeg()
    reign_grow(out_pcd)