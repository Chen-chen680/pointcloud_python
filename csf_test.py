import laspy
import CSF
import numpy as np

from utils.utils import las_visualization

def csf_lvbo():
    # -----------------------读取las点云----------------------
    las = laspy.read('./data/eight/cloud_18148ef5.las')
    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    # ----------------------布料模拟滤波----------------------
    csf = CSF.CSF()
    # 参数设置
    csf.params.bSloopSmooth = False  # 粒子设置为不可移动
    csf.params.cloth_resolution = 0.1  # 布料网格分辨率
    csf.params.rigidness = 3  # 布料刚性参数
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.03  # 点云与布料模拟点的距离阈值
    csf.params.interations = 500  # 最大迭代次数
    # more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # 地面点索引列表

    non_ground = CSF.VecInt()  # 非地面点索引列表
    csf.do_filtering(ground, non_ground)  # 执行滤波

    # ----------------------保存为las文件-----------------------
    # 地面点
    ground_las = laspy.LasData(las.header)
    ground_las.points = las[np.array(ground)]  # *****
    ground_las.write("./data/eight/ground.las")
    # 非地面点

    non_ground_las = laspy.LasData(las.header)
    non_ground_las.points = las[np.array(non_ground)]  # ****
    non_ground_las.write("./data/eight/non_ground.las")


if __name__ == '__main__':
    csf_lvbo()