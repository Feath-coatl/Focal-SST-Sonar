#该模块主要功能是生成一些实验结果示意图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import pyvista as pv
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import random
import sklearn.cluster
import sklearn.svm
from sklearn.svm import OneClassSVM
from scipy.spatial import KDTree
import copy
import time
import math
import re
import ast
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams, PersLandscapeExact
from persim.landscapes import plot_landscape

from knn_classifier import KNNClassifier, GridSearch
from scipy.spatial import distance_matrix
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#sklearn
import sklearn
import sklearn.feature_extraction
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from header import read_frame_pts, map_value_to_rgb, drop_null_data, sample_distance, MyTimer, sigmoid, import_from_datasets
import gudhi as gd

#open3d风格，对点云进行可视化
def visualize_ptcloud_o3d(pcd, window_name="Custom Window", other=None):


    # 创建Visualizer
    vis = o3d.visualization.O3DVisualizer()
    vis.create_window(window_name=window_name, width=800, height=600)

    # 添加几何体
    pcd.normals = o3d.utility.Vector3dVector([]) #取消法向量设置
    vis.add_geometry(pcd)
    if other != None:
        other.normals = o3d.utility.Vector3dVector([]) 
        vis.add_geometry(other)

    # 设置背景颜色（RGB值，范围0到1）
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.2, 0.3]  # 设置为深蓝色背景

    # 渲染图像
    vis.run()
    vis.destroy_window()

#绘制多个几何体
def visualize_multi_geometrics(canvas=[], window_name='result', labels=[]):
    # 创建Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)

    # 添加几何体
    for item in canvas:
        if isinstance(item, o3d.geometry.PointCloud):
            item.normals = o3d.utility.Vector3dVector([])
        vis.add_geometry(item)

    #添加文字信息
    for label in labels:
        vis.add_geometry(label)
        #vis.add_3d_label(label[0], label[1])
        pass
    # 设置背景颜色（RGB值，范围0到1）
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.2, 0.3]  # 设置为深蓝色背景

    # 渲染图像
    vis.run()
    vis.destroy_window() 

#绘制多个几何体和文字标签
def visualize_app(canvas=[], window_name='result', labels=[]):
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3) * 10)

    # 创建窗口和场景
    app = gui.Application.instance
    app.initialize()
    window = gui.Application.instance.create_window(window_name, 800, 600)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # 添加点云到场景中
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    i=0
    # 添加几何体
    for item in canvas:
        i = i+1
        if isinstance(item, o3d.geometry.PointCloud):
            item.normals = o3d.utility.Vector3dVector([])
        scene_widget.scene.add_geometry(str(i), item, material)

    # 设置场景背景色
    scene_widget.scene.set_background([0.1, 0.2, 0.3, 0.95])  # 白色背景

    # 创建 3D 标签
    for label in labels:
        tmp = scene_widget.add_3d_label(label[0], label[1])
        tmp.color = gui.Color(1.0, 1.0, 1.0)
        #tmp.scale(2, center=(0, 0, 0)) 

    # 设置摄像机参数
    bounds = pcd.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60, bounds, bounds.get_center())

    # 运行应用
    app.run()

#将numpy.array的点云数据转换为o3d.PointCloud类型
def transfer_nparr_to_ptcloud(points, color_min=1, color_max=1, with_intensity=True):
    #提取向量信息
    pts_np = points
    if not isinstance(points, np.ndarray):
        pts_np = np.asarray(points)
    xyz = pts_np[:,:-2]
    intensity = pts_np[:,-2]
    #
    #对原始强度值进行操作
    #强度值标准化
    mscaler = sklearn.preprocessing.StandardScaler()
    intensity_re = intensity.reshape(-1,1)
    intensity_std = mscaler.fit_transform(intensity_re)
    intensity_std = intensity_std.flatten()
    #
    #将强度值映射为RGB颜色
    colors = np.zeros((xyz.shape[0], 3))
    min_val = intensity_std.min()*color_min
    max_val = intensity_std.max()*color_max
    for i in range(intensity_std.shape[0]):
        colors[i] = map_value_to_rgb(intensity_std[i], min_val, max_val)
    #将标准化的强度值用normals保存下来
    normals = np.zeros((intensity.shape[0], 3))
    normals[:,0] = intensity
    #使用向量创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

#生成ROC曲线图
def draw_roc_curve():
    # 生成示例数据
    np.random.seed(42)
    n_samples = 500
    X = np.random.rand(n_samples, 2)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 70%负类, 30%正类

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # 使用随机森林分类器
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # 计算预测概率
    y_scores = clf.predict_proba(X_test)[:, 1]  # 获取正类概率

    # 计算 FPR, TPR 和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # 计算 AUC
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

#kd-tree优化的最远点采样算法
def farthest_point_sampling_with_intensity(pcd, k):
    """
    基于 KD-Tree 的最远点采样算法，支持 PointCloud 格式并保留强度值。
    
    参数:
        pcd (open3d.geometry.PointCloud): 输入点云对象。
        k (int): 需要采样的点数。
        
    返回:
        sampled_pcd (open3d.geometry.PointCloud): 采样后的点云对象。
    """
    # 提取点云的坐标和强度值
    points = np.asarray(pcd.points)  # 点的坐标
    intensities = np.asarray(pcd.colors)  # 假设强度值保存在颜色属性中
    
    num_points = points.shape[0]
    if k > num_points:
        raise ValueError("采样点数 k 必须小于等于点云总数")

    # 初始化
    sampled_indices = []         # 采样点的索引
    distances = np.full(num_points, np.inf)  # 初始化每个点的最小距离为正无穷
    selected_index = np.random.randint(num_points)  # 随机选择一个初始点
    sampled_indices.append(selected_index)

    # 使用 KD-Tree 构建点云索引
    kd_tree = KDTree(points)

    # 最远点采样过程
    for _ in range(1, k):
        # 更新所有点到当前采样点集的最小距离
        distances = np.minimum(
            distances,
            np.linalg.norm(points - points[selected_index], axis=1)
        )
        
        # 选择最远的点
        selected_index = np.argmax(distances)
        sampled_indices.append(selected_index)

    # 提取采样点的坐标和强度值
    sampled_points = points[sampled_indices]
    sampled_intensities = intensities[sampled_indices]

    # 创建新的点云对象
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_intensities)

    return sampled_pcd

#计算一个点云的关键点
def extract_keypoints(pcd, salient_radius=0.1, non_max_radius=0.2, gamma_21=0.975, gamma_32=0.975):
    keypoint_num = 50
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    keypoints_indices = []

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
        keypoints_indices = indices
        #提取关键点
        #keypoints = pcd.select_by_index(indices)
        keypoints = farthest_point_sampling_with_intensity(pcd, k=keypoint_num)
        ####################################################
    elif len(keypoints.points)>keypoint_num:
        #关键点个数过多则随机采样
        points = np.asarray(keypoints.points)
        selectd_indices = random.sample(range(len(points)), keypoint_num)
        selected_points = points[selectd_indices]
        keypoints_indices = indices

        keypoints.points = o3d.utility.Vector3dVector(selected_points)

    # 可视化点云与关键点
    # 设置关键点颜色为红色
    keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    pcd.normals = o3d.utility.Vector3dVector([]) #取消法向量设置
    keypoints.normals = o3d.utility.Vector3dVector([]) #取消法向量设置

    #获取点云颜色
    colors = np.asarray(pcd.colors)
    # 将指定索引的点的颜色设置为红色
    for idx in keypoints_indices:
        colors[idx] = [1.0, 0.0, 0.0]  # 红色的 RGB 值为 [1.0, 0.0, 0.0]
    # 更新点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([pcd,keypoints], window_name="ISS Keypoints