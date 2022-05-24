import open3d as o3d
import laspy
import numpy as np
from pclpy import pcl
import os
import random
import glob
import math
from sklearn.cluster import KMeans
import sklearn
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import shutil


def read_las(las_path):
    '''读取las文件'''
    las = laspy.read(las_path)
    return las


def las2pcd(las_cloud):
    '''将las格式的文件转化成pcd格式'''
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose() # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose() # 获取颜色
    pcd = o3d.geometry.PointCloud() # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz) # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color) # 给pcd对象添加颜色
    return pcd


def uniform_down_sample(pcd):
    '''均值下采样'''
    print("原始点云中点的个数为：", np.asarray(pcd.points).shape[0])
    print("每5个点来降采样一个点")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    print("下采样之后点的个数为：", np.asarray(uni_down_pcd.points).shape[0])
    return uni_down_pcd


def las_visualization(las_cloud):
    """las格式点云可视化"""
    pcd = las2pcd(las_cloud)
    pcd_visualization(pcd)


def pcd_visualization(pcd):
    """pcd格式点云可视化"""
    o3d.visualization.draw_geometries([pcd])


def euclidean_distance(one_sample, X):
    """计算欧式距离"""
    # 将one_sample转换为一纬向量
    one_sample = one_sample.reshape(1, -1)
    # 把X转换成一维向量
    X = X.reshape(X.shape[0], -1)
    # 这是用来确保one_sample的尺寸与X相同
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


def EuclideanSeg_xyzrgb(pcd_path, ClusterTolerance=0.5, MinClusterSize=10):
    '''欧式距离分割，聚类，实现岩石结构面识别'''
    # -----------------------创建保存的文件夹----------------------------
    save_dir = os.path.join(os.path.dirname(pcd_path), 'euclidean_xyzrgb')
    create_dir_w(save_dir)
    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZRGB()
    pcl.io.loadPCDFile(pcd_path, pc)
    # ------------------------------欧式聚类----------------------------------
    ec = pcl.segmentation.EuclideanClusterExtraction.PointXYZRGB()
    tree = pcl.search.KdTree.PointXYZRGB()
    ec.setInputCloud(pc)                      # 输入点云
    ec.setSearchMethod(tree)                  # 设置搜索方式为kdtree
    ec.setClusterTolerance(ClusterTolerance)                 # 设置近邻搜索的搜索半径为1m
    ec.setMaxClusterSize(1000000000)               # 一个聚类需要的最大点数
    ec.setMinClusterSize(MinClusterSize)                  # 一个聚类需要的最小点数
    clusters = pcl.vectors.PointIndices()     # 构建保存分割结果的索引向量
    ec.extract(clusters)                      # 获取分割索引
    # ----------------------------输出相关信息---------------------------------
    max_cluster_count = max([len(c.indices) for c in clusters])
    # print('聚类的个数为：', len(clusters))
    # print('聚类结果中的最大点数为：', max_cluster_count)
    # -----------------------保存聚类结果到本地文件夹---------------------------
    extract = pcl.filters.ExtractIndices.PointXYZRGB()
    it = 0
    result_list = []
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
        result_list.append(cloud_inliers)
        it += 1
    return result_list


def reign_grow_xyzrbga(path,
                       MinClusterSize=10,
                       Neighbours=5,
                       SmoothnessThreshold=5 / 180 * math.pi,
                       CurvatureThreshold=2,
                       ResidualThreshold=0.5):
    """使用颜色、坐标实现区域生长算法，进行岩石分割"""
    # -----------------------创建保存的文件夹----------------------------
    save_dir = os.path.join(os.path.dirname(path), 'region_xyzrgb')
    create_dir_w(save_dir)
    '''区域生长法'''
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
    rg.setMaxClusterSize(5000000)                               # 一个聚类需要的最大点数
    rg.setMinClusterSize(MinClusterSize)                        # 一个聚类需要的最小点数
    rg.setNumberOfNeighbours(Neighbours)                        # 搜索的邻域点的个数
    rg.setSmoothnessThreshold(SmoothnessThreshold)              # 设置平滑阈值，法线差值阈值
    rg.setCurvatureThreshold(CurvatureThreshold)                # 设置表面曲率的阈值
    rg.setResidualThreshold(ResidualThreshold)                  # 设置残差阈值
    clusters = pcl.vectors.PointIndices()                       # 构建保存分割结果的索引向量
    rg.extract(clusters)                                        # 获取分割索引
    # ----------------------------输出相关信息---------------------------------
    max_cluster_count = max([len(c.indices) for c in clusters])
    # print('聚类的个数为：', len(clusters))
    # print('聚类结果中的最大点数为：', max_cluster_count)
    # -----------------------保存聚类结果到本地文件夹---------------------------
    extract = pcl.filters.ExtractIndices.PointXYZRGBA()
    it = 0
    result_list = []
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
        result_list.append(cloud_inliers)
        it += 1
    return result_list


def reign_grow_xyz(path,
                   MinClusterSize=10,
                   Neighbours=5,
                   SmoothnessThreshold=5 / 180 * math.pi,
                   CurvatureThreshold=2,
                   ResidualThreshold=0.5,
                   fore_clusters_num=10,
                   total_clusters_num=20):
    '''使用坐标实现区域生长法，进行岩石结构面提取'''
    # -----------------------创建保存的文件夹----------------------------
    save_dir = os.path.join(os.path.dirname(path), 'region_xyz')
    create_dir_w(save_dir)
    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZ()
    pcl.io.loadPCDFile(path, pc)
    # -------------------------法向量和表面曲率估计----------------------------
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normals_estimation.setInputCloud(pc)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(0.35)
    normals_estimation.compute(normals)
    # ------------------------------区域生长----------------------------------
    rg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()       # 创建区域生长分割对象
    rg.setInputCloud(pc)                                        # 输入点云
    rg.setInputNormals(normals)                                 # 输入法向量
    rg.setMaxClusterSize(5000000)                               # 一个聚类需要的最大点数
    rg.setMinClusterSize(MinClusterSize)                        # 一个聚类需要的最小点数
    rg.setNumberOfNeighbours(Neighbours)                        # 搜索的邻域点的个数
    rg.setSmoothnessThreshold(SmoothnessThreshold)              # 设置平滑阈值，法线差值阈值
    rg.setCurvatureThreshold(CurvatureThreshold)                # 设置表面曲率的阈值
    rg.setResidualThreshold(ResidualThreshold)                  # 设置残差阈值
    clusters = pcl.vectors.PointIndices()                       # 构建保存分割结果的索引向量
    rg.extract(clusters)                                        # 获取分割索引
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_inliers.xyz)
        result_list.append(pcd)
        it += 1
        # todo 1
    after_seg_connect_small_clusters(result_list, total_clusters_num, save_dir, method='region_grow')


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


def main_sloop(pcd):
    '''坡道滤波算法主程序'''
    grid_step = 2  # 格网边长
    elevation_difference_threshold = 0.5  # 高差阈值
    slope_threshold = 0.1  # 坡度阈值
    ground_cloud, other_cloud = slope_filter(pcd, grid_step, elevation_difference_threshold, slope_threshold)
    return ground_cloud, other_cloud


def connect_pcd(pcd_list):
    """
    拼接多个pcd点云为一个pcd点云
    :param pcd_list: 需要拼接的pcd文件列表
    :return: 拼接后的pcd对象
    """
    connected_pcd = o3d.geometry.PointCloud()
    for idx, pcd in enumerate(pcd_list):
        if idx == 0:
            connect_pcd_array = np.asarray(pcd.points)
        else:
            temp_points = np.asarray(pcd.points)
            connect_pcd_array = np.concatenate((connect_pcd_array, temp_points))
    connected_pcd.points = o3d.utility.Vector3dVector(connect_pcd_array)
    return connected_pcd


def k_means_clusters_after_seg(pcd, save_dir, method, num_clusters=5):
    """处理用其他方法分割后的过小的部分，使用k均值聚类进行二次聚类"""
    create_dir_w(save_dir)
    # ----------------------------加载点云数据---------------------------------
    data1 = np.asarray(pcd.points)
    # ----------------------------进行数据训练---------------------------------
    # 第一种方式
    # 标准化
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data1)
    # 预估计流程
    estimator = KMeans(n_clusters=num_clusters)
    kmeans_model = estimator.fit(data_new)
    labels = kmeans_model.labels_
    scors = sklearn.metrics.calinski_harabasz_score(data_new, labels)
    print(scors, save_dir, method)

    y_pred = estimator.predict(data_new)
    # -------------------------使用open3d获取点云分类结果------------------------
    result_list = []
    for cluster in range(num_clusters):
        idx = np.where(y_pred == cluster)[0]  # 获取第一类的索引(第2，3，，，n类获取方式与此相同)
        cloud = pcd.select_by_index(idx)  # 根据索引提取位于第一类中的点云
        # o3d.io.write_point_cloud(os.path.join(save_dir, method + str(cluster) +".pcd"), cloud)  # 保存点云
        result_list.append(cloud)
    return result_list


def after_seg_connect_small_clusters_delete(pcd_list, fore_clusters_num, total_clusters_num, save_dir, method):
    '''
    读取分割后的初始点云，进行预处理，将小的点云使用K均值聚类为大的点云
    :param pcd_list: pcd的列表
    :param fore_cluster_num: 保留前几个聚类
    :param total_cluster_num: 总的聚类的个数
    :param save_dir: 保存的位置
    :return: 返回合并后的pcd
    '''
    #-------------------将点按从小到大排列------------------------
    total_pcd_idx_num_list = []
    for idx, pcd in enumerate(pcd_list):
        points_num = np.asarray(pcd.points).shape[0]  # 读取点的个数
        total_pcd_idx_num_list.append([idx, points_num])  # 返回pcd中的idx以及点的个数
    total_pcd_idx_num_list.sort(reverse=True, key=lambda idx_num_list: idx_num_list[1])  # 按照点的个数，进行排序
    #-----------------将点云分为两组----------------------------------
    fore_pcd_idx_num_list = total_pcd_idx_num_list[0:fore_clusters_num:]
    after_pcd_idx_num_list = total_pcd_idx_num_list[fore_clusters_num::]
    fore_pcd_list, after_pcd_list = [], []
    #------------------------不需要拼接的直接粘贴到只当位置-----------------------------
    for pcd_idx_num in fore_pcd_idx_num_list:
        fore_pcd_list.append(pcd_list[pcd_idx_num[0]])
    for pcd_idx_num in after_pcd_idx_num_list:
        after_pcd_list.append(pcd_list[pcd_idx_num[0]])
    #----------------使用Kmeans拼接的小的集群--------------------------
    connected_pcd = connect_pcd(after_pcd_list)
    result_list = k_means_clusters_after_seg(connected_pcd, save_dir, method, num_clusters=total_clusters_num - fore_clusters_num)
    for idx, cloud in enumerate(result_list):
        o3d.io.write_point_cloud(os.path.join(save_dir, method + '_' + str(idx)) + '.pcd', cloud)


def after_seg_connect_small_clusters(pcd_list, total_clusters_num, save_dir, method):
    '''提取后的结构面点云，将其保存下来'''
    connected_pcd = connect_pcd(pcd_list)
    result_list = k_means_clusters_after_seg(connected_pcd, save_dir, method, num_clusters=total_clusters_num)
    for idx, cloud in enumerate(result_list):
        o3d.io.write_point_cloud(os.path.join(save_dir, method + '_' + str(idx)) + '.pcd', cloud)


def create_dir_w(dir):
    '''判断文件夹是否存在，若存在，则删除该文件夹，若不存在，则删除该文件夹'''
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else:
        shutil.rmtree(dir, True)
        os.makedirs(dir)

def k_means_clusters_test_score():
    '''k均值聚类的ch指标计算'''
    num_clusters = 7
    pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_wait_seg.pcd'
    save_dir = os.path.join(os.path.dirname(pcd_path), 'kmeans')
    # create_dir_w(save_dir)
    # ----------------------------加载点云数据---------------------------------
    pcd = o3d.io.read_point_cloud(pcd_path)
    data1 = np.asarray(pcd.points)
    # ----------------------------进行数据训练---------------------------------
    # 第一种方式
    # 标准化
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data1)
    # 预估计流程
    estimator = KMeans(n_clusters=num_clusters)
    kmeans_model = estimator.fit(data_new)
    labels = kmeans_model.labels_
    scors = sklearn.metrics.calinski_harabasz_score(data_new, labels)
    print(scors)
    # y_pred = estimator.predict(data_new)
    # -------------------------使用open3d获取点云分类结果------------------------
    # for cluster in range(num_clusters):
    #     idx = np.where(y_pred == cluster)[0]  # 获取第一类的索引(第2，3，，，n类获取方式与此相同)
    #     cloud = pcd.select_by_index(idx)  # 根据索引提取位于第一类中的点云
    #     o3d.io.write_point_cloud(os.path.join(save_dir, 'kmeans_' +str(cluster) + '.pcd'), cloud)


def k_means_clusters(pcd_path, num_clusters=20):
    '''K均值聚类实现岩石结构面的识别'''
    save_dir = os.path.join(os.path.dirname(pcd_path), 'kmeans')
    create_dir_w(save_dir)
    # ----------------------------加载点云数据---------------------------------
    pcd = o3d.io.read_point_cloud(pcd_path)
    data1 = np.asarray(pcd.points)
    # ----------------------------进行数据训练---------------------------------
    # 第一种方式
    # 标准化
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data1)
    # 预估计流程
    estimator = KMeans(n_clusters=num_clusters)
    kmeans_model = estimator.fit(data_new)
    labels = kmeans_model.labels_
    scors = sklearn.metrics.calinski_harabasz_score(data_new, labels)
    print(scors, pcd_path)

    y_pred = estimator.predict(data_new)
    # -------------------------使用open3d获取点云分类结果------------------------
    for cluster in range(num_clusters):
        idx = np.where(y_pred == cluster)[0]  # 获取第一类的索引(第2，3，，，n类获取方式与此相同)
        cloud = pcd.select_by_index(idx)  # 根据索引提取位于第一类中的点云
        o3d.io.write_point_cloud(os.path.join(save_dir, 'kmeans_' +str(cluster) + '.pcd'), cloud)

def ch_metrics():
    '''计算聚类后的ch指标'''
    import numpy as np
    from sklearn import metrics
    from sklearn.cluster import KMeans

    k_means_clusters(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_wait_seg.pcd', num_clusters=5)
    dataframe = np.random.randint(0, 50, size=(200, 10))
    # 以kmeans聚类方法为例
    kmeans_model = KMeans(n_clusters=3, random_state=1).fit(dataframe)
    labels = kmeans_model.labels_
    score = metrics.calinski_harabasz_score(dataframe, labels)
    print(score)

#  获取xyz方向的最大值和最小值
def get_min_max_3d(cloud):
    '''得到点云x,y,z的最大、最小值'''
    points = np.asarray(cloud.points)
    min_pt_0 = np.amin(points, axis=0)
    max_pt_0 = np.amax(points, axis=0)

    # min_pt_1 = np.amin(points, axis=1)
    # max_pt_1 = np.amax(points, axis=1)

    # min_pt_2 = np.amin(points, axis=2)
    # max_pt_2 = np.amax(points, axis=2)
    return min_pt_0, max_pt_0

def get_trans_point(cloud):
    '''得到坐标转换的点'''
    points = np.asarray(cloud.points)
    point = points[0]
    return point

def plane_fit():
    '''使用RANSAC算法实现平面拟合(测试代码)'''
    base_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data'
    huapo_list = ['eight', 'bai', 'tang']
    houzhui_list = ['region_xyz', 'euclidean_xyz', 'dbscan', 'kmeans']
    fig = plt.figure(figsize=(10, 8))
    for huapo in huapo_list:
        pcd_basepath = os.path.join(base_path, huapo, 'region_xyz')
        pcd_path_list = glob.glob(os.path.join(pcd_basepath, '*.pcd'))
        for idx,pcd_path in enumerate(pcd_path_list):
            pcd = o3d.io.read_point_cloud(pcd_path)
            if idx == 0:
                trans_xmin, trans_ymin, trans_zmin = get_trans_point(pcd)
            plane_model, inliers = pcd.segment_plane(distance_threshold=1,
                                                     ransac_n=50,
                                                     num_iterations=1000)
            [a, b, c, d] = plane_model
            # print(pcd_path, f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            # equation = f"{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0"
            pts_min, pts_max = get_min_max_3d(pcd)
            x_min, y_min, z_min = pts_min[0], pts_min[1], pts_min[2]
            x_max, y_max = pts_max[0], pts_max[1]
            x = np.linspace(x_min - trans_xmin, x_max - trans_xmin, 1000)
            y = np.linspace(y_min - trans_ymin, y_max - trans_ymin, 1000)
            X,Y = np.meshgrid(x, y)
            exec(f"Z = ((-{a:.2f}*(X- trans_xmin) - {b:.2f}*(Y - trans_ymin) - {d:.2f}) / {c:.2f}) + {c:.2f}* trans_zmin")
            exec("ax" + str(idx) + "= fig.gca(projection='3d')")
            exec("ax"+ str(idx) + ".plot_surface(X, Y, Z)")
        fig.show()
        print(huapo)
        plt.clf()

def plane_fit_1():
    '''使用RANSAC拟合提取的结构面，并绘制出其结构面'''
    base_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data'
    huapo_list = ['eight', 'bai', 'tang']
    houzhui_list = ['region_xyz', 'euclidean_xyz', 'dbscan', 'kmeans']
    fig = plt.figure(figsize=(10, 8))
    for huapo in huapo_list:
        pcd_basepath = os.path.join(base_path, 'tang', 'region_xyz')
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
            # if idx == 0:
            #     record_x_min, record_y_min, record_z_min = x_min, y_min, z_min
            #     record_x_max, record_y_max, record_z_max = x_max, y_max, z_max
            # else:
            #     if x_min < record_x_min:
            #         record_x_min = x_min
            #     if y_min < record_y_min:
            #         record_y_min = y_min
            #     if z_min < record_z_min:
            #         record_z_min = z_min
            #     if x_max > record_x_max:
            #         record_x_max = x_max
            #     if y_max > record_y_max:
            #         record_y_max = y_max
            #     if z_max > record_z_max:
            #         record_z_max = z_max

            x = np.linspace(x_min, x_max, 1000)
            y = np.linspace(y_min, y_max, 1000)
            X,Y = np.meshgrid(x, y)
            print((f"Z = ((-{a:.2f}*(X) - {b:.2f}*(Y) - {d:.2f}) / {c:.2f})"))
            exec(f"Z = ((-{a:.2f}*(X) - {b:.2f}*(Y) - {d:.2f}) / {c:.2f})")
            exec("ax" + str(idx) + "= fig.gca(projection='3d')")
            exec("ax"+ str(idx) + ".plot_surface(X, Y, Z)")
            # exec("ax" + str(idx) + ".set_xlim(record_x_min, record_x_max)")
            # exec("ax" + str(idx) + ".set_ylim(record_y_min, record_y_max)")
            # exec("ax" + str(idx) + ".set_zlim(record_z_min, record_z_max)")
            # fig.set_ylim(record_y_min, record_y_max)
            # fig.set_zlim(record_z_min, record_z_max)
        fig.show()

def axis_trans():
    '''坐标转换'''
    pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_wait_seg.pcd'
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
    pts_min = np.amin(points, axis=0)
    points[:, 0] = points[:, 0] - np.mean(points[:, 0])
    points[:, 1] = points[:, 1] - np.mean(points[:, 1])
    # points[:, 2] = points[:, 2] - np.mean(points[:, 2])
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_wait_seg_trans.pcd', pcd)

if __name__ == '__main__':
    # eight_las_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    # eight_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd'
    # bai_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd'
    # tang_las_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las'
    # tang_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_uniform_downsample.pcd'
    #
    # eight_wait_seg_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_wait_seg.pcd'
    # bai_wait_seg_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_wait_seg.pcd'
    # tang_wait_seg_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_wait_seg_trans.pcd'
    # wai_seg_path_list = [tang_wait_seg_path]
    # wai_seg_path_list = [eight_wait_seg_path, bai_wait_seg_path, tang_wait_seg_path]
    # down_sample_path_list = [eight_pcd_path, bai_pcd_path, tang_pcd_path]
    #
    # huapo_list = ['eight', 'bai', 'tang']
    # houzhui_list = ['region_xyz', 'euclidean_xyz', 'dbscan', 'kmeans']
    #
    # las_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'

    # plane_fit()
    plane_fit_1()
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\region_xyz')
    # ---------------- wait seg show-------------------------
    # for idx, path in enumerate(down_sample_path_list):
    #     print(path)
    #     if idx == 1:
    #         pcd = o3d.io.read_point_cloud(path)
    #         pcd.paint_uniform_color([0,1,0])
    #         o3d.visualization.draw_geometries([pcd])

    # ----------------------统计滤波-------------------------------
    # pcd_path_list = [bai_pcd_path]
    # for pcd_path in pcd_path_list:
    #     pcd = o3d.io.read_point_cloud(pcd_path)
    #     sor_pcd, out_pcd = statistics_remove_points(pcd)
    #     sor_pcd.paint_uniform_color([0,1,0])
    #     out_pcd.paint_uniform_color([1,0,0])
    #     print(pcd_path)
    #     o3d.visualization.draw_geometries([sor_pcd, out_pcd])

    # --------------------------坡度滤波算法-----------------------------
    # pcd_path_list = [eight_pcd_path, tang_pcd_path, bai_pcd_path]
    # for idx, pcd_path in enumerate(pcd_path_list):
    #     if idx == 2:
    #         pcd = o3d.io.read_point_cloud(pcd_path)
    #         ground_pcd, unground_pcd = main_sloop(pcd)
    #         unground_pcd.paint_uniform_color([1,0,0])
    #
    #         ground_pcd.paint_uniform_color([0,1,0])
    #         print(pcd_path)
    #         o3d.visualization.draw_geometries([ground_pcd, unground_pcd])
    #     else:
    #         continue
    #---------------------full process---------------------
    # pcd = o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd')  # uniform downsample
    # for pcd_path in down_sample_path_list:
    #     pcd = o3d.io.read_point_cloud(pcd_path)
    #     sor_cloud, out_cloud = statistics_remove_points(pcd)  # 统计滤波
    #     ground_cloud, nonground_cloud = main_sloop(sor_cloud)  # 地面滤波
    #     o3d.io.write_point_cloud(os.path.join(os.path.dirname(pcd_path), r'wait_seg.pcd'), ground_cloud)  # 保存文件

    # for pcd_path in wai_seg_path_list:
    #     reign_grow_xyz(pcd_path, total_clusters_num=7, fore_clusters_num=3, MinClusterSize=1, Neighbours=10)
    #     EuclideanSeg_xyz(pcd_path, total_cluters_num=7, fore_clusters_num=3, min_cluster_size=1, cluster_tolerance=2)
    #     dbscan_cluster(pcd_path, min_points=1, eps=2, fore_clusters_num=3, total_clusters_num=7)
    #     k_means_clusters(pcd_path, num_clusters=7)

    # dir_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data'
    # for huapo in huapo_list:
    #     for method in houzhui_list:
    #         watch_path = os.path.join(dir_path, huapo, method)
    #         print(watch_path)
    #         bacth_pcd_viewer(watch_path)
    #         break


    # reign_grow_xyzrbga(eight_wait_seg_path)
    # EuclideanSeg_xyzrgb(eight_wait_seg_path)

    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\dbscan')
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\euclidean_xyz')
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\euclidean_xyzrgb')
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\regisonxyzrgb')
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\regionxyz')

    # pcd = o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_wait_seg.pcd')
    # o3d.visualization.draw_geometries([pcd])

    # min_cut_seg(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_wait_seg.pcd')

    #  -------------------------Kmeans_cluster--------------------------
    # pcd = o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_wait_seg.pcd')
    # k_means_clusters(pcd, save_dir=r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\kmeans', num_clusters=20)
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\kmeans')

    # --------------------connect small points after seg-----------------------------
    # fore_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight'
    # for base_path in houzhui_list:
    #     full_path = os.path.join(fore_path, base_path)
    #     after_seg_connect_small_clusters(full_path, 10, 20)

    # ----------------------different method result view--------------------
    # fore_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight'
    # for base_path in houzhui_list:
    #     print(base_path)
    #     full_path = os.path.join(fore_path, base_path, 'after_seg')
    #     bacth_pcd_viewer(full_path)

    # --------------------------不同方法综合2022.4.25日10点41分--------------------------
    # pcd_path_list = [eight_wait_seg_path, bai_wait_seg_path, tang_wait_seg_path]
    # pcd_path_list = [eight_wait_seg_path]
    # for pcd_path in pcd_path_list:
    #     EuclideanSeg_xyz(pcd_path, cluster_tolerance=0.5, min_cluster_size=1,fore_clusters_num=10, total_cluters_num=20)
    #     reign_grow_xyz(pcd_path,
    #                    MinClusterSize=5,
    #                    Neighbours=5,
    #                    SmoothnessThreshold=5 / 180 * math.pi,
    #                    CurvatureThreshold=2,
    #                    ResidualThreshold=0.5,
    #                    fore_clusters_num=10,
    #                    total_clusters_num=20)
    #     dbscan_cluster(pcd_path, eps=0.5, min_points=1)
    #     k_means_clusters(pcd_path, num_clusters=20)

    # path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight'
    # bacth_pcd_viewer(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\region_xyz')
    # for method in houzhui_list:
    #     bacth_pcd_viewer(os.path.join(path, method))
    # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd')])
    # k_means_clusters_test_score()
    # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_wait_seg.pcd')])
    # axis_trans()