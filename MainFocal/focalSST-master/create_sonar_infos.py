import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
ROOT_PATH = Path('MainFocal/focalSST-master/data/sonar')
# 点云文件夹
POINTS_DIR = ROOT_PATH / 'points'
# 划分文件文件夹
IMAGESETS_DIR = ROOT_PATH / 'ImageSets'

# 类别映射：TXT中第5列的值 -> OpenPCDet中的类别名称
# 仅处理这里定义的类别，背景(3)和水面(4)会被自动忽略
CLASS_MAPPING = {
    1: 'Box',  # 请根据实际情况修改名称，如 'Box'
    2: 'Diver'   # 请根据实际情况修改名称，如 'Diver'
}
# ===========================================

def get_box_from_points(points):
    """
    使用 PCA (主成分分析) 计算带旋转的 3D Bounding Box
    Args:
        points: (N, 3) numpy array [x, y, z]
    Returns:
        box: [x, y, z, dx, dy, dz, heading]
    """
    if points.shape[0] < 30:
        # 点太少无法计算 PCA，返回全0
        return np.zeros(7, dtype=np.float32)

    # 1. 计算 XY 平面的主成分 (PCA) 以确定 Heading
    points_xy = points[:, :2]
    mean_xy = np.mean(points_xy, axis=0)
    # 归一化中心
    centered_xy = points_xy - mean_xy
    # 计算协方差矩阵
    cov = np.cov(centered_xy.T)

    try:
        evals, evecs = np.linalg.eig(cov)
        # 检查特征值是否为 NaN 或 负数 (计算误差)
        if np.any(np.isnan(evals)) or np.any(evals < 0):
            return np.zeros(7, dtype=np.float32)
            
        sort_indices = np.argsort(evals)[::-1]
        principal_axis = evecs[:, sort_indices[0]]
        heading = np.arctan2(principal_axis[1], principal_axis[0])
    except Exception:
        # 如果 PCA 崩溃（例如所有点重合），返回无效框
        return np.zeros(7, dtype=np.float32)
    
    # 2. 将点云旋转到“正”方向，以便计算长宽
    c, s = np.cos(heading), np.sin(heading)
    rotation_matrix = np.array([[c, s], [-s, c]]) # 旋转矩阵 (World -> Local)
    
    # 旋转点云
    points_local_xy = np.dot(centered_xy, rotation_matrix.T)
    
    # 3. 计算局部坐标系下的边界 (Min/Max)
    min_xy = np.min(points_local_xy, axis=0)
    max_xy = np.max(points_local_xy, axis=0)
    
    # Z 轴不旋转，直接取 Min/Max
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    
    # 4. 计算 Box 尺寸 (dx, dy, dz)
    # dx: 沿主轴的长度 (Length)
    # dy: 垂直主轴的宽度 (Width)
    # dz: 高度 (Height)
    l = max_xy[0] - min_xy[0]
    w = max_xy[1] - min_xy[1]
    h = z_max - z_min
    
    # 5. 计算 Box 中心 (x, y, z)
    # 局部坐标系下的中心
    center_local_xy = (min_xy + max_xy) / 2
    
    # 转换回世界坐标系
    # R_inv = R.T (正交矩阵)
    # World = Mean + Local * R_inv
    # 这里 rotation_matrix 是 [[c, s], [-s, c]]
    # 它的逆(转置)是 [[c, -s], [s, c]]
    R_inv = rotation_matrix.T
    center_world_xy = mean_xy + np.dot(center_local_xy, R_inv)
    
    center_z = (z_min + z_max) / 2
    
    return np.array([center_world_xy[0], center_world_xy[1], center_z, l, w, h, heading], dtype=np.float32)

def process_split(split_name):
    txt_list_file = IMAGESETS_DIR / f'{split_name}.txt'
    
    if not txt_list_file.exists():
        print(f"警告: 未找到 {txt_list_file}，跳过 {split_name} 集。")
        return
        
    with open(txt_list_file, 'r') as f:
        sample_id_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
    infos = []
    
    print(f"正在处理 {split_name} 集，共 {len(sample_id_list)} 帧...")
    
    for sample_idx in tqdm(sample_id_list):
        txt_path = POINTS_DIR / f'{sample_idx}.txt'
        if not txt_path.exists():
            print(f"错误: 找不到点云文件 {txt_path}")
            continue
            
        # 读取点云: [x, y, z, intensity, class]
        points_all = np.loadtxt(str(txt_path), dtype=np.float32)
        if points_all.ndim == 1:
            points_all = points_all.reshape(1, -1)

        # === 坐标系对齐 ===
        # 原始: [x, y, z, i, c] -> 变换: [y, -x, z, i, c]
        points_temp = points_all.copy()
        points_all[:, 0] = points_temp[:, 1]  # New X = Old Y
        points_all[:, 1] = -points_temp[:, 0] # New Y = -Old X
        # =======================================================
            
        # 初始化标注列表
        gt_boxes = []
        gt_names = []
        
        # 遍历我们在 CLASS_MAPPING 中定义的目标类别 (1, 2)
        for class_id, class_name in CLASS_MAPPING.items():
            # 提取该类别的点
            mask = (points_all[:, 4] == class_id)
            target_points = points_all[mask]
            
            if target_points.shape[0] > 0:
                # 判断点数是否小于30，若不足则跳过并打印信息
                if target_points.shape[0] < 30:
                    #print(f"警告: 样本 {sample_idx}.txt 中类别 {class_name} 的点数为 {target_points.shape[0]}（小于30），跳过该目标框生成。")
                    continue
                # 每帧每类只有一个实例，直接计算整体的 Box
                box7 = get_box_from_points(target_points[:, :3])
                if box7[3] * box7[4] * box7[5] < 0.1:
                    print(f"警告: 样本 {sample_idx} 产生极微小体积框，已跳过。")
                    continue
                    
                # 2. 检查是否有 NaN (非常重要)
                if np.any(np.isnan(box7)):
                    print(f"警告: 样本 {sample_idx} 产生NaN框，已跳过。")
                    continue

                gt_boxes.append(box7)
                gt_names.append(class_name)
        
        # 构建 Info 字典
        info = {
            'point_cloud': {
                'lidar_idx': sample_idx,
            }
        }
        
        if len(gt_boxes) > 0:
            info['annos'] = {
                'name': np.array(gt_names),
                'gt_boxes_lidar': np.array(gt_boxes), # (N, 7)
                # OpenPCDet 有时需要 num_points_in_gt，这里暂时不填或填假数据
                # 'num_points_in_gt': np.array([pts.shape[0]...]) 
            }
        else:
            # 没有目标的空帧 (背景帧)
            info['annos'] = {
                'name': np.zeros(0),
                'gt_boxes_lidar': np.zeros((0, 7))
            }
            
        infos.append(info)
        
    # 保存 .pkl
    output_path = ROOT_PATH / f'sonar_infos_{split_name}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f"成功生成: {output_path}")

if __name__ == '__main__':
    if not POINTS_DIR.exists():
        print(f"错误: 数据路径不存在 {POINTS_DIR}")
        exit()
        
    process_split('train')
    process_split('val')