import numpy as np
import open3d as o3d
import copy
from matplotlib import pyplot as plt
import glob
import os
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

eight_las = r'.\data\eight\cloud_18148ef5.las'
tang_las = r'.\data\tang\cloud_-73352f21.las'

# 在点云上添加分类标签
def draw_labels_on_model(pcl, labels):
    cmap = plt.get_cmap("tab20")
    pcl_temp = copy.deepcopy(pcl)
    max_label = labels.max()
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    pcl_temp.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcl_temp], window_name="可视化分类结果",
                                      width=800, height=800, left=50, top=50,
                                      mesh_show_back_face=False)

# 计算欧氏距离
def euclidean_distance(one_sample, X):
    # 将one_sample转换为一纬向量
    one_sample = one_sample.reshape(1, -1)
    # 把X转换成一维向量
    X = X.reshape(X.shape[0], -1)
    # 这是用来确保one_sample的尺寸与X相同
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances

def k_means(pcd, save_dir, num_clusters=5):
    # ----------------------------加载点云数据---------------------------------
    data1 = np.asarray(pcd.points)
    # ----------------------------进行数据训练---------------------------------
    # 第一种方式
    # 标准化
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data1)
    # 预估计流程
    estimator = KMeans(n_clusters=num_clusters)
    estimator.fit(data_new)
    y_pred = estimator.predict(data_new)

    # -------------------------使用open3d获取点云分类结果------------------------
    for cluster in range(num_clusters):
        idx = np.where(y_pred == cluster)[0]  # 获取第一类的索引(第2，3，，，n类获取方式与此相同)
        cloud = pcd.select_by_index(idx)  # 根据索引提取位于第一类中的点云
        o3d.io.write_point_cloud(os.path.join(save_dir, "111.pcd"), cloud)  # 保存点云


def las_viewer(las_cloud):
    '''las数据可视化'''
    xyz = np.vstack((las_cloud.x, las_cloud.y, las_cloud.z)).transpose()  # 获取坐标
    color = np.vstack((las_cloud.red / 65025, las_cloud.green / 65025, las_cloud.blue / 65025)).transpose()  # 获取颜色
    pcd = o3d.geometry.PointCloud()  # 创建pcd对象
    pcd.points = o3d.utility.Vector3dVector(xyz)  # 给pcd对象添加点
    pcd.colors = o3d.utility.Vector3dVector(color)  # 给pcd对象添加颜色
    down_pcd = my_uniform_down_sample(pcd) # 降采样
    return down_pcd # 返回降采样之后的点云数据
    # o3d.visualization.draw_geometries([pcd])

def my_uniform_down_sample(pcd):
    print("原始点云中点的个数为：", np.asarray(pcd.points).shape[0])
    # o3d.visualization.draw_geometries([pcd])
    print("每5个点来降采样一个点")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=1000)
    print("下采样之后点的个数为：", np.asarray(uni_down_pcd.points).shape[0])
    return uni_down_pcd

def rgb_choice(path):
    with open(path, 'r', encoding='gbk') as f:
        contents = f.readlines()
    new_contents = []
    for index, rgb in enumerate(contents):
        if index == 0:
            continue
        temp = rgb.replace('\n', '').split(',')
        for idx, number in enumerate(temp):
            temp[idx] = float(number)
        new_contents.append(temp)
    return new_contents

def bacth_pcd_viewer(path):
    '''las数据可视化'''
    rgb_list = rgb_choice('rgb_choice1.CSV')
    path_list = glob.glob(os.path.join(path, '*.pcd'))

    for idx, pcd_path in enumerate(path_list):
        exec('pcd' + str(idx) + '= o3d.io.read_point_cloud(pcd_path)')
        exec('pcd' + str(idx) + '.paint_uniform_color(' + str([round(random.uniform(0,1), 1), round(random.uniform(0,1), 1), round(random.uniform(0,1), 1)]) +')')

    #------------------------------------创建绘图-------------------------------------------
    execution = 'o3d.visualization.draw_geometries(['
    for i in range(len(path_list)):
        execution = execution + 'pcd' + str(i) + ','
    execution = execution[0: -1] + '])'
    print(execution)
    exec(execution)

if __name__ == "__main__":
    #  加载点云
    # las = laspy.read(r'.\data\八字门\cloud_18148ef5.las')
    # pcd = las_viewer(las)
    #
    # points = np.asarray(pcd.points)
    # rgb = np.asarray(pcd.colors)
    # origin_points = np.concatenate([points, rgb], 1)
    # origin_points = pd.DataFrame(origin_points, columns=['x', 'y', 'z', 'r', 'g', 'b'])
    # origin_points.to_csv('origin_points.csv')
    #
    # o3d.visualization.draw_geometries([pcd], window_name="可视化原始点云",
    #                                   width=800, height=800, left=50, top=50,
    #                                   mesh_show_back_face=False)
    # # # 执行K-means聚类
    # clf = Kmeans(k=4)
    # labels = clf.predict(points)
    # labels = np.reshape(labels, [labels.shape[0], 1])
    # new_points = np.concatenate([points, labels], 1)
    # new_points = pd.DataFrame(new_points, columns=['x', 'y', 'z', 'label'])
    # new_points.to_csv('result_points.csv')

    bacth_pcd_viewer(r'./eucli')