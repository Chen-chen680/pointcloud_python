import laspy
import numpy as np
import open3d as o3d
from pclpy import pcl
import os
import time

eight_las = r'.\data\eight\cloud_18148ef5.las'
eight_pcd = r'.\data\eight\cloud_18148ef5.pcd'
tang_las = r'.\data\tang\cloud_-73352f21.las'


def main():
    las = laspy.read(eight_las)
    las_viewer(las)


def las_viewer(las_cloud):
    '''las数据可视化'''
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose() # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose() # 获取颜色
    pcd = o3d.geometry.PointCloud() # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz) # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color) # 给pcd对象添加颜色
    o3d.visualization.draw_geometries([pcd])
    # my_uniform_down_sample(pcd)
    # o3d.visualization.draw_geometries([pcd])


def las2pcd(las_cloud):
    """las数据可视化"""
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose() # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose() # 获取颜色
    pcd = o3d.geometry.PointCloud() # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz) # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color) # 给pcd对象添加颜色
    return pcd

def my_uniform_down_sample(pcd):
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    Ransac(uni_down_pcd)
    # o3d.visualization.draw_geometries([uni_down_pcd],
    #                                   window_name="均匀下采样",
    #                                   width=1200, height=800,
    #                                   left=50, top=50)

def Ransac(pcd):
    # pcd = o3d.io.read_point_cloud("R1.pcd")
    plane_model, inliers = pcd.segment_plane(distance_threshold=1,
                                             ransac_n=50,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def test():
    from pclpy import pcl
    import math

    # ------------------------------读取点云----------------------------------
    pc = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile('street_thinned.pcd', pc)

    # -------------------------法向量和表面曲率估计----------------------------
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZRGBA_Normal()
    normals_estimation.setInputCloud(pc)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(0.35)
    normals_estimation.compute(normals)
    # ------------------------------区域生长----------------------------------
    rg = pcl.segmentation.RegionGrowing.PointXYZRGBA_Normal()  # 创建区域生长分割对象
    rg.setInputCloud(pc)  # 输入点云
    rg.setInputNormals(normals)  # 输入法向量
    rg.setMaxClusterSize(1000000)  # 一个聚类需要的最大点数
    rg.setMinClusterSize(10)  # 一个聚类需要的最小点数
    rg.setNumberOfNeighbours(15)  # 搜索的邻域点的个数
    rg.setSmoothnessThreshold(5 / 180 * math.pi)  # 设置平滑阈值，法线差值阈值
    rg.setCurvatureThreshold(5)  # 设置表面曲率的阈值
    rg.setResidualThreshold(1)  # 设置残差阈值
    clusters = pcl.vectors.PointIndices()  # 构建保存分割结果的索引向量
    rg.extract(clusters)  # 获取分割索引
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
        file_name = "RegionGrong_" + str(it) + ".pcd"
        pcl.io.savePCDFile(file_name, cloud_inliers)
        it += 1
    # ---------------------可视化聚类结果----------------------------
    colored_cloud = pcl.PointCloud.PointXYZRGBA()
    colored_cloud = rg.getColoredCloud()
    viewer = pcl.visualization.PCLVisualizer("RegionGrowing")
    viewer.setBackgroundColor(0, 0, 0)
    viewer.addPointCloud(colored_cloud, "RegionGrowing cloud")
    viewer.setPointCloudRenderingProperties(0, 1, "RegionGrowing cloud")
    while not viewer.wasStopped():
        viewer.spinOnce(10)


def statistics_remove_points(pcd, nb_neighbors=30, std_ratio=2):
    '''
    :param pcd: pcd数据格式
    :param nb_neighbors: 多少个临近点
    :param std_ratio: 多少个std
    :return: 返回内部点、离群点
    '''
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    sor_cloud = pcd.select_by_index(ind)
    out_cloud = pcd.select_by_index(ind, invert=True)
    return sor_cloud, out_cloud

def show_las_with_intensity(las_path=r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\cloud_-620047b9.las'):
    # ---------------------读取las格式的点云-----------------------
    cloud = pcl.PointCloud.PointXYZRGB()
    pcl.io.loadPCDFile(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd', cloud)
    print('读取点云的点数为：', cloud.size())
    print('前10个点的坐标为为:', np.asarray(cloud.xyz)[:10])
    # ------------------从las中获取xyz坐标和强度-------------------
    # XYZI = np.vstack((pc.x, pc.y, pc.z, pc.intensity)).transpose()
    # cloud = pcl.PointCloud.PointXYZI.from_array(XYZI)
    # print('读取点云的点数为：', cloud.size())
    # print('前10个点的坐标为为:', np.asarray(cloud.xyz)[:10])
    # ----------------------可视化强度------------------------
    viewer = pcl.visualization.PCLVisualizer("3D viewer")
    viewer.setBackgroundColor(1, 1, 1)
    color = pcl.visualization.PointCloudColorHandlerGenericField.PointXYZI(cloud, "intensity")

    viewer.addPointCloud(cloud, color, "sample cloud")
    viewer.setPointCloudRenderingProperties(0, 1, "sample cloud")
    # viewer.addCoordinateSystem(1)
    # viewer.initCameraParameters()
    while not viewer.wasStopped():
        viewer.spinOnce(10)

def show_with_normal():
    # 创建XYZ格式的点云
    cloud = pcl.PointCloud.PointXYZ()
    # 读取点云数据
    reader = pcl.io.PCDReader()
    reader.read(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd', cloud)
    n = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    # n.setViewPoint(0, 0, 0)  # 设置视点，默认为（0，0，0）
    n.setInputCloud(cloud)
    n.setNumberOfThreads(6)
    n.setKSearch(10)  # 点云法向计算时，需要所搜的近邻点大小
    # n.setRadiusSearch(0.03)  # 半径搜素
    normals = pcl.PointCloud.Normal()
    n.compute(normals)  # 开始进行法向计
    # for normal in normals.points:
    #     print(normal.normal_x, normal.normal_y, normal.normal_z)


    d = np.hstack((cloud.xyz, normals.normals))  # 拼接字段
    cloud_with_normal = pcl.PointCloud.PointNormal.from_array(d)
    # ----------------------可视化法向量的变化------------------------
    viewer = pcl.visualization.PCLVisualizer("3D viewer")
    viewer.setBackgroundColor(1, 1, 1)
    color = pcl.visualization.PointCloudColorHandlerGenericField.PointNormal(cloud_with_normal, "curvature")

    viewer.addPointCloud(cloud_with_normal, color, "sample cloud")
    viewer.setPointCloudRenderingProperties(0, 1, "sample cloud")
    # viewer.addCoordinateSystem(1)
    # viewer.initCameraParameters()
    while not viewer.wasStopped():
        viewer.spinOnce(10)



def compute_std_div_mean(pcd):
    xyz = np.asarray(pcd.points)
    # ------------------计算点云的均值、方差和标准差----------------------------
    [mean_x, mean_y, mean_z] = np.mean(xyz, axis=0)  # 计算输入点云x、y、z坐标的均值
    [var_x, var_y, var_z] = np.var(xyz, axis=0, ddof=1)
    [std_x, std_y, std_z] = np.std(xyz, axis=0, ddof=1)  # 计算输入点云x、y、z坐标的标准差(PCL中计算的是n-1标准差，这里借鉴一下)
    # --------------------------输出计算结果-----------------------------------
    print('均值为：', [mean_x, mean_y, mean_z])
    print('方差为：', [var_x, var_y, var_z])
    print('标准差为：', [std_x, std_y, std_z])
    return [std_x / mean_x, std_y / mean_y, std_z / mean_z]

def get_total_cloudpoint_num():
    las_bai_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\cloud_-620047b9.las'
    las_tang_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las'
    las_eight_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    las_path_list = [las_eight_path, las_bai_path, las_tang_path]

    eight_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd'
    tang_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_uniform_downsample.pcd'
    bai_pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd'
    pcd_path_list = [eight_pcd_path, bai_pcd_path, tang_pcd_path]
    for idx, las_path in enumerate(las_path_list):
        if idx == 0:
            las_name = 'eight'
            continue
        elif idx == 1:
            las_name = 'bai'
        else:
            las_name = 'tang'
        las = laspy.read(las_path)
        pcd = las2pcd(las)
        print(las_name, compute_std_div_mean(pcd))
        # print(las_name, las.header.generating_software, las.header.creation_date, las.header.version, las.header.point_count)
        # print(las_name, las.header.x_max- las.header.x_min,  las.header.y_max- las.header.y_min, las.header.z_max- las.header.z_min)
        # print(las_name)

        # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(las_path)])

def batch_view():
    las_bai_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\cloud_-620047b9.las'
    las_tang_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las'
    las_eight_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    las_path_list = [las_eight_path, las_bai_path, las_tang_path]

    pcd_eight_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd'
    pcd_bai_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd'
    pcd_tang_pach = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_uniform_downsample.pcd'
    pcd_path_list = [pcd_eight_path, pcd_bai_path, pcd_tang_pach]

    for las_path in las_path_list:
        las = laspy.read(las_path)
        print(las_path)
        las_viewer(las)


    for pcd_path in pcd_path_list:
        pcd = o3d.io.read_point_cloud(pcd_path)
        print(pcd_path)
        o3d.visualization.draw_geometries([pcd])
        import sklearn
        sklearn.metrics

def get_all_point_num():
    las_bai_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\cloud_-620047b9.las'
    las_tang_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las'
    las_eight_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    las_path_list = [las_eight_path, las_bai_path, las_tang_path]
    for las_path in las_path_list:
        las_head = laspy.open(las_path)
        print(las_path,las_head.header.point_count)

def show_with_xyz():

    # ----------------------读取点云------------------------
    cloud = pcl.PointCloud.PointXYZRGB()
    pcl.io.loadPCDFile(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\bai_uniform_downsample.pcd', cloud)
    print('读取点云的点数为：', cloud.size())
    print('前10个点的坐标为为:', np.asarray(cloud.xyz)[:10])
    # -----------------------可视化------------------------
    viewer = pcl.visualization.PCLVisualizer("3D viewer")
    viewer.setBackgroundColor(1, 1, 1)
    color = pcl.visualization.PointCloudColorHandlerGenericField.PointXYZRGB(cloud, "z")

    viewer.addPointCloud(cloud, color, "sample cloud")
    viewer.setPointCloudRenderingProperties(0, 1, "sample cloud")
    # viewer.addCoordinateSystem(1)
    # viewer.initCameraParameters()
    while not viewer.wasStopped():
        viewer.spinOnce(10)


if __name__ == '__main__':
    las_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    las = laspy.read(las_path)
    print(las_path)
    las_viewer(las)

    pcd_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd'
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(pcd_path)
    o3d.visualization.draw_geometries([pcd])


    # las_bai_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\bai\cloud_-620047b9.las'
    # las_tang_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las'
    # las_eight_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\cloud_18148ef5.las'
    # pcd_tang_path = r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.pcd'
    #
    # las = laspy.read(las_eight_path)
    # print('read complete')
    # pcd = las2pcd(las)
    # print('开始降采样')
    # pcd = pcd.uniform_down_sample(every_k_points=50)
    # print('降采样结束')
    # o3d.io.write_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\co
    # de\project1\data\eight\eight_uniform_downsample', pcd)

    # pcd = o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\eight\eight_uniform_downsample.pcd')
    # o3d.visualization.draw_geometries([pcd])

    # get_total_cloudpoint_num()

    # show_with_normal()
    # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\tang_uniform_downsample.pcd')])
    # las_cloud = laspy.read(r'C:\Users\12517\Desktop\ultimate_paper\code\project1\data\tang\cloud_-73352f21.las')
    # las_viewer(las_cloud)
    # get_all_point_num()
    # get_total_cloudpoint_num()

    # batch_view()
    # show_with_xyz()
    # show_las_with_intensity()
    # show_with_normal()