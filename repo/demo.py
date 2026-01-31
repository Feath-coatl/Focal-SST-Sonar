import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os
import math
import re
import ast
import random
from datetime import datetime
import scipy.special
import sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import sklearn.svm
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from collections import Counter

from ripser import ripser
from persim import plot_diagrams
import scipy

#wasserstein & knn_classifier
from knn_classifier import sample_distance, KNNClassifier, GridSearch
from header import read_frame_pts, map_value_to_rgb, drop_null_data, read_frame, visualize_ptcloud_o3d
from cluster import cluster_kmeans
#sklearn
import sklearn
from sklearn.cluster import KMeans
import sklearn.feature_extraction
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing

#用于生成实验结果示意图
from generator import draw_roc_curve, DBSCAN_cluster, draw_data_per_prec, draw_ransac, draw_fps, draw_DBSCAN, draw_PD_PB_PL, farthest_point_sampling_with_intensity, draw_PH_diff, ph_time_count, \
    calculate_cluster_params, calculate_class_distance, gather_dataset_info, data_process, draw_res, draw_ransac_res, draw_plt, draw_vr_com, draw_vr_com_3d

#显示三维点云数据
def show_frame(data, classid=3, title="other", size=4):
    fig = plt.figure()
    #print(data)
    #遍历各个分组
    #设置位置和颜色
    color = "red"
    if classid == 1:
        color = 'saddlebrown'
        pos = 221
    elif classid == 2:
        color = 'forestgreen'
        pos = 222
    elif classid == 3:
        color = 'mediumorchid'
        pos = 223
    elif classid == 4:
        color = 'deepskyblue'
        pos = 223
    #绘制散点
    #x,y,z = zip(*data)
    axes = fig.add_subplot(111, projection='3d')
    #axes.scatter(x,y,z,c=color)
    axes.scatter(data['x'], data['y'], data['z'])
    plt.title(title)
    plt.show()


#在可视化中突出一下特征点
def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres

#初始化(将采样，标准化, 提取特征点)
def preprocessing_frame(pts_np, voxel_size=0.05, pts_type="other"):
    #提取向量信息
    xyz = pts_np[:,:-2]
    intensity = pts_np[:,-2]
    #
    #对原始强度值进行操作
    #强度值标准化
    mscaler = sklearn.preprocessing.StandardScaler()
    intensity_re = intensity.reshape(-1,1)
    intensity_std = mscaler.fit_transform(intensity_re)
    intensity_std = intensity_std.flatten()
    #print("标准化强度值", intensity_std)
    #
    #将强度值映射为RGB颜色
    colors = np.zeros((xyz.shape[0], 3))
    min_val = intensity_std.min()
    max_val = intensity_std.max()*0.8
    if max_val <= min_val:
        max_val = 1.0
        min_val = 0.0
    for i in range(intensity_std.shape[0]):
        colors[i] = map_value_to_rgb(intensity_std[i], min_val, max_val)
    #将标准化的强度值用normals保存下来
    normals = np.zeros((intensity.shape[0], 3))
    normals[:,0] = intensity
    #print(colors)
    #使用向量创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    pcd.normals = o3d.utility.Vector3dVector(normals)
    #下采样
    #pcd = pcd.voxel_down_sample(voxel_size)
    #去质心
    centroid = pcd.get_center()
    #print("点云质心为：", centroid)
    pts_non_center = pcd.points - centroid
    pcd.points = o3d.utility.Vector3dVector(pts_non_center)
    #print(pcd.points)
    #返回ndarray
    xyz_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)
    normals_np = np.asarray(pcd.normals) #实际上是强度值
    #可视化
    #取消normals设置
    pcd.normals = o3d.utility.Vector3dVector([])
    #visualize_ptcloud_o3d(pcd, window_name=pts_type)
    #o3d.visualization.draw_geometries([pcd],
    #                                  width = 1200,
    #                                  height = 800,
    #                                  left = 50,
    #                                  top = 50)
    
    ###############################
    #test:提取点云特征点
    keypoint_num = 64
    #if pts_type == 'woodenframe':
    #    keypoint_num = 64
    #elif pts_type == 'frogman':
    #    keypoint_num = 50
    #elif pts_type == 'surface':
    #    keypoint_num = 32
    #elif pts_type == 'other':
    #    keypoint_num = 32
    
    #用最远点采样降采样到对应点数
    keypoints = pcd
    if len(pcd.points) >= keypoint_num:
        keypoints = farthest_point_sampling_with_intensity(pcd, keypoint_num)
    #使用ISS提取关键点
    """
    使用 Open3D 的 ISS 关键点提取功能。
    参数：
        point_cloud: Open3D 点云对象
        gamma_21: 控制线性显著性的阈值
        gamma_32: 控制平面显著性的阈值
        min_neighbors: 每个点需要的最少邻居数（过滤低密度点）
    返回：
        关键点点云
    """
    # 配置 ISS 关键点检测参数
    #keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, 
    #                                                        salient_radius=,
    #                                                        non_max_radius=1.5,
    #                                                        gamma_21=1.5,
    #                                                        gamma_32=1.5)
    '''
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

    #如果关键点个数过少，则采用基于FPFH的方法提取关键点
    if len(keypoints.points)<keypoint_num:
        #计算FPFH特征值
        #计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #提取fpfh特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        #选择特征点
        # 4. 计算每个点的特征描述子的L2范数
        norms = np.linalg.norm(fpfh.data, axis=0)
        # 5. 选择最具代表性的n个点（特征描述子范数最大的点）
        #设置关键点数量
    
        indices = np.argsort(norms)[-keypoint_num:]
        #提取关键点
        keypoints = pcd.select_by_index(indices)
        ####################################################
    elif len(keypoints.points)>keypoint_num:
        #关键点个数过多则随机采样
        points = np.asarray(keypoints.points)
        selectd_indices = random.sample(range(len(points)), keypoint_num)
        selected_points = points[selectd_indices]

        keypoints.points = o3d.utility.Vector3dVector(selected_points)
    '''


    #可视化
    #print("keypoints/total:", len(keypoints.points), ':', len(pcd.points))
    #keypoints.paint_uniform_color([1.0, 0.0, 0.0])  # 将特征点颜色设置为红色
    #pcd.paint_uniform_color([0.5, 0.5, 0.5]) #降原始点云设置为灰色
    #o3d.visualization.draw_geometries([keypoints], 
    #                                  window_name=pts_type,
    #                                  width=800, height=600,
    #                                  point_show_normal=False)
    #返回坐标和强度信息
    return xyz_np, normals_np[:, 0], keypoints
    #return xyz_np, colors_np, keypoints

#------------聚类测试---------------#
#用kmeans方法对点云数据进行聚类
def cluster_kmeans(sonar=None):
    #1.读取数据
    data = read_frame_pts("../../data/sonar/test1/data.pts")
    sonar = pd.DataFrame(data, columns=["x", "y", "z", "intensity", "class"])
    show_frame(sonar)
    sonar = sonar[(sonar["class"] != 4.0)]
    show_frame(sonar)
    sonar = sonar.iloc[:,0:-2]
    #print(sonar)
    #2.预估器流程
    estimator = KMeans(n_clusters=3)
    estimator.fit(sonar)
    y_predict = estimator.predict(sonar)
    #3.模型评估
    #用轮廓系数进行评估
    score = sklearn.metrics.silhouette_score(sonar, y_predict)
    print(y_predict)
    print(score)
    #4.可视化
    labels = estimator.labels_
    labels = pd.DataFrame(labels, columns=['class'])
    show_data = sonar.copy()
    show_data['class'] = labels

    show_frame(show_data)


#---------------------------------#
#用体素方法计算点云的体积
def calculate_volume(points):
    x_min, y_min, z_min = np.amin(points, axis=0)
    x_max, y_max, z_max = np.amax(points, axis=0)

    step = 0.05
    row = np.ceil((x_max-x_min)/step)
    col = np.ceil((y_max-y_min)/step)
    cel = np.ceil((z_max-z_min)/step)

    if row == 0 or col == 0 or cel == 0:
        return 0
    m = np.zeros((int(row), int(col), int(cel)))

    for i in range(len(points)):
        rid = np.floor((points[i][0]-x_min)/step)
        cid = np.floor((points[i][1]-y_min)/step)
        eid = np.floor((points[i][2]-z_min)/step)
        m[int(rid), int(cid), int(eid)] += 1

    num = np.count_nonzero(m)
    return num*np.power(step,3) 

#计算和可视化持续同调
def persistent_homology_demo(points, keypoints, classid=3, pts_type="other"):
    #输入点云，计算其持续同调，并以持久性图/条形码可视化
    #print("原始点云图像：", classid)
    #show_frame(points, classid, pts_type)
    ################################
    #考虑加入补全点云的操作，大概步骤是先生成三角网格，再调用库补全，再转换为散点……有点怪
    ################################
    #计算持续同调
    #dgms = ripser(points, maxdim=2, thresh=0.25)['dgms']
    #目前统计H0, H1, H2
    dgms = ripser(keypoints, maxdim=2, distance_matrix=False, warning_flag=False)['dgms']
    #maxdim指定计算H0,H1,...Hn的最大维度n，n大于1时就巨慢
    #thresh指定滤流的最大半径
    plot_diagrams(dgms, show=True, title=pts_type)
    return dgms

#字符串处理
def line_to_list(line):
    #print(line)
    #删除数字后的多余空格
    line = re.sub(r'(\b\w+\s+)\s+(\])', r'\1\2', line)
    #删除数字前的多余空格
    line = re.sub(r'\[\s+', '[', line)
    #删除逗号后的多余空格
    line = line.replace()