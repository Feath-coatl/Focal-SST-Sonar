import pandas as pd
import os
from tqdm import tqdm
import ot
import numpy as np
import open3d as o3d
import re
import ast
import time
#公共头文件

# 定义 Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#自定义计时器
class MyTimer:
    def __init__(self, title=''):
        self.start_time = time.perf_counter()
        self.title = title
    
    def time_up(self):
        end_time=time.perf_counter()
        elapsed_time = end_time - self.start_time
        #print(f'过程{self.title}共耗时{elapsed_time}秒')
        return elapsed_time

#自定义距离函数，采用Wasserstein距离
def sample_distance(a,b):
    #wasserstein_distance(u_values, v_values, u_weights, v_weights)
    #计算wasserstein距离时，应该将分布的下标放在value处，将分布的权重放在weight处
    #大胆猜测一下就是x在前，y在后
    #分别统计一下H0和H1、H2的wasserstein距离
    #a = np.array(a)
    #b = np.array(b)
    emd = []
    '''
    for i in range(3):
        u = np.array(a[i])
        if len(u)==0:
            continue
        u0 = u[:, 0]
        u1 = u[:, 1]
        v = np.array(b[i])
        if len(v)==0:
            continue
        v0 = v[:, 0]
        v1 = v[:, 1]
        dist0 = scipy.stats.wasserstein_distance(u0, v0)
        dist1 = scipy.stats.wasserstein_distance(u1, v1)
        dist = np.sqrt(dist0**2 + dist1**2)
    '''
    for i in range(3):
        u = np.array(a.iloc[i])
        if len(u)==0:
            u = np.array([[0,0]])
            #continue
        n1 = u.shape[0]
        w1 = np.full(n1, 1/n1)

        v = np.array(b.iloc[i])
        if len(v)==0:
            v = np.array([[0,0]])
            #continue
        n2 = v.shape[0]
        w2 = np.full(n2, 1/n2)

        M = ot.dist(u, v, metric='euclidean')  # 欧几里得距离矩阵
        dist = ot.emd2(w1, w2, M)

        emd.append(dist)

    #print("\n计算距离：", a, '-', b, '\n')
    #u_values = a[:,0]
    #v_values = b[:,0]
    #u_weights = a[:,1]
    #v_weights = b[:,1]

    #u_weights[u_weights[:] == 0] = 0.001
    #v_weights[v_weights[:] == 0] = 0.001
    #emd = scipy.stats.wasserstein_distance(u_values, v_values)
    ##print(emd)
    return np.sum(emd)

#读取一帧数据(效率太低)
def read_frame(file_path):
    columns=["x", "y", "z", "intensity", "class"]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=columns)
    # 删除包含 NaN 值的行
    df = df.dropna()
    # 删除至少有 3 个值为 0 的行
    df = df[df.apply(lambda row: (row == 0).sum() < 3, axis=1)]

    return df

#读取一帧数据(.pts格式)
def read_frame_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    #提取点坐标
    points = []
    for line in lines:
        #忽略注释
        if line.startswith("#"):
            continue
        #提取坐标
        values = line.split()
        x = float(values[0])
        y = float(values[1])
        z = float(values[2])
        intensity = float(values[3])
        cla = float(values[4])
        points.append([x,y,z,intensity,cla])
    
    #返回list形式
    return points

def map_value_to_rgb(value, min_val=0, max_val=1, cmap_name='viridis'):
    #最小值为蓝色，最大值为红/紫色
    range = max_val - min_val
    if range <= 0 :
        range = 1.0
    if value<min_val:
        return [0,0,0]
    r = (value-min_val)/range
    step = range/5
    idx = int(r*5)
    h = (idx+1)*step+min_val
    m = idx*step+min_val
    local_r = (value-m)/(h-m)
    if value<min_val:
        return [0,0,0]
    if value>max_val:
        return [1,1,1]
    if (idx==0):
        return [0,local_r,1]
    if (idx==1):
        return [0,1,1-local_r]
    if (idx==2):
        return [local_r,1,0]
    if (idx==3):
        return [1,1-local_r,0]
    if (idx==4):
        return [1,0,local_r]
    
#文本处理，删除无用的点云数据
def drop_null_data(work_dir=''):
    for root,dirs,files in os.walk(work_dir):
        #创建一个进度条
        total = len(files)
        pbar = tqdm(range(total), desc="Processing", bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for file in files:
            file_path = os.path.join(root,file)
            output_file = '../res/res.txt'
            df = pd.read_csv(file_path, sep='\s+', header=None)
            # 删除包含 NaN 值的行
            df = df.dropna()

            # 删除至少有 3 个值为 0 的行
            df = df[df.apply(lambda row: (row == 0).sum() < 3, axis=1)]
            df.to_csv(file_path, index=False, header=False, sep=' ')
            pbar.update(1)

#open3d风格，对点云进行可视化
def visualize_ptcloud_o3d(pcd, window_name="Custom Window", other=None):
    # 创建Visualizer
    vis = o3d.visualization.Visualizer()
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

    #字符串处理
def line_to_list(line):
    #print(line)
    #删除数字后的多余空格
    line = re.sub(r'(\b\w+\s+)\s+(\])', r'\1\2', line)
    #删除数字前的多余空格
    line = re.sub(r'\[\s+', '[', line)
    #删除逗号后的多余空格
    line = line.replace(", ", ",")
    #将多个空格替换为'，'
    line = re.sub(r'\s+', ',', line)
    #将换行符去除
    line = line.replace('\n', '],[')
    #将.替换为.0
    line = re.sub(r'(?<=\b0)\.(?=\D|\b)', '.0', line)
    #将inf替换为0.0
    line = line.replace('inf','2.0')
    #转换为列表
    return ast.literal_eval(line)

#从存储的datasets文件中读取和处理信息
def import_from_datasets(save_file_name=""):
    data = pd.read_csv(save_file_name)
    #df = pd.DataFrame([data], columns=["points_num", "avg_intensity", "var_intensity", "avg_radius", "avg_distance", "avg_curvature", "volume", "ph_H0", "ph_H1", "class"])
    data0 = data["ph_H0"]
    data1 = data["ph_H1"]
    data2 = data["ph_H2"]
    H0 = []
    H1 = []
    H2 = []
    #处理H0
    total0 = len(data0)
    pbar0 = tqdm(range(total0), desc="importing H0", bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for line in data0:
        record = line_to_list(line)
        H0.append(record)
        pbar0.update(1)
    data["ph_H0"] = H0
    #处理H1
    total1 = len(data1)
    pbar1 = tqdm(range(total1), desc="importing H1", bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for line in data1:
        record = line_to_list(line)
        H1.append(record)
        pbar1.update(1)
    data["ph_H1"] = H1
    #处理H2
    total2 = len(data2)
    pbar2 = tqdm(range(total2), desc="importing H2", bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for line in data2:
        record = line_to_list(line)
        H2.append(record)
        pbar2.update(1)
    data["ph_H2"] = H2
    print("\n----数据加载完成<se.2>----\n")

    return data