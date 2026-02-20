import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
ROOT_PATH = Path('data/sonar')
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

# 鲁棒拟合参数：用于抑制离群噪声、保证框覆盖主体点
MIN_INLIER_POINTS = 20
ROBUST_ZSCORE_THRESHOLD = 3.5
MIN_INLIER_RATIO = 0.55
MIN_COVERAGE_RATIO = 0.88
INSIDE_MARGIN = 0.03
MIN_BOX_SIZE = 0.05
MAX_REMOVAL_RATIO = 0.05
# ===========================================


def robust_filter_points(points, sample_idx=None, class_name=None):
    """
    对单目标点云做鲁棒离群点过滤。

    设计目标：
    1) 去除与主体距离很远的少量误标噪声点，避免GT框异常变大；
    2) 避免过度过滤导致目标主体被删掉。
    """
    if points.shape[0] < MIN_INLIER_POINTS:
        return points, 0

    # Step-1: 基于坐标MAD的鲁棒z-score过滤
    median_xyz = np.median(points, axis=0)
    mad_xyz = np.median(np.abs(points - median_xyz), axis=0)
    robust_scale = 1.4826 * mad_xyz + 1e-3
    robust_dist = np.sqrt(np.sum(((points - median_xyz) / robust_scale) ** 2, axis=1))
    mask_mad = robust_dist <= ROBUST_ZSCORE_THRESHOLD

    # Step-2: 基于XY径向IQR过滤远离主体中心的噪声点
    center_xy = np.median(points[:, :2], axis=0)
    radial_dist = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    q1, q3 = np.percentile(radial_dist, [25, 75])
    iqr = max(q3 - q1, 1e-3)
    radial_thresh = q3 + 1.5 * iqr
    mask_radial = radial_dist <= radial_thresh

    # 综合掩码
    mask = mask_mad & mask_radial

    n_points = points.shape[0]
    max_remove = int(np.floor(n_points * MAX_REMOVAL_RATIO))
    min_keep_by_ratio = n_points - max_remove
    min_keep = max(MIN_INLIER_POINTS, int(points.shape[0] * MIN_INLIER_RATIO))
    min_keep = max(min_keep, min_keep_by_ratio)

    # 若双重过滤过于激进，则回退为“仅剔除最异常的最多5%点”
    if np.sum(mask) < min_keep:
        if max_remove <= 0:
            filtered_points = points
        else:
            radial_norm = radial_dist / (np.median(radial_dist) + 1e-3)
            anomaly_score = robust_dist + radial_norm
            keep_indices = np.argsort(anomaly_score)[:min_keep_by_ratio]
            filtered_points = points[np.sort(keep_indices)]
    else:
        filtered_points = points[mask]

    removed_count = int(n_points - filtered_points.shape[0])
    if removed_count > 0:
        removed_ratio = removed_count / n_points
        if sample_idx is not None and class_name is not None:
            print(f"[去噪] 样本 {sample_idx} 类别 {class_name}: 剔除 {removed_count}/{n_points} ({removed_ratio:.2%})")
        else:
            print(f"[去噪] 剔除 {removed_count}/{n_points} ({removed_ratio:.2%})")

    return filtered_points, removed_count


def points_inside_box_ratio(points, box7, margin=INSIDE_MARGIN):
    """计算点云落在3D旋转框内的比例，用于质量校验。"""
    if points.shape[0] == 0:
        return 0.0

    cx, cy, cz, dx, dy, dz, heading = box7
    dx = max(float(dx), MIN_BOX_SIZE)
    dy = max(float(dy), MIN_BOX_SIZE)
    dz = max(float(dz), MIN_BOX_SIZE)

    centered = points - np.array([cx, cy, cz], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)  # World -> Local
    local_xy = np.dot(centered[:, :2], rot.T)

    inside_x = np.abs(local_xy[:, 0]) <= (dx * 0.5 + margin)
    inside_y = np.abs(local_xy[:, 1]) <= (dy * 0.5 + margin)
    inside_z = np.abs(centered[:, 2]) <= (dz * 0.5 + margin)

    return float(np.mean(inside_x & inside_y & inside_z))

def get_box_from_points(points, sample_idx=None, class_name=None):
    """
    使用 PCA (主成分分析) 计算带旋转的 3D Bounding Box
    Args:
        points: (N, 3) numpy array [x, y, z]
    Returns:
        box: [x, y, z, dx, dy, dz, heading]
    """
    if points.shape[0] < 3:
        return np.zeros(7, dtype=np.float32)

    # 0. 先做鲁棒离群点过滤，降低误标噪声对框的影响
    points_inlier, _ = robust_filter_points(points, sample_idx=sample_idx, class_name=class_name)
    if points_inlier.shape[0] < 3:
        return np.zeros(7, dtype=np.float32)

    # 1. 计算 XY 平面的主成分 (PCA) 以确定 Heading
    points_xy = points_inlier[:, :2]
    mean_xy = np.mean(points_xy, axis=0)
    # 归一化中心
    centered_xy = points_xy - mean_xy
    # 计算协方差矩阵
    cov = np.cov(centered_xy.T)

    try:
        # 协方差矩阵是实对称矩阵，使用eigh更稳定
        evals, evecs = np.linalg.eigh(cov)
        # 检查特征值是否为 NaN 或 负数 (计算误差)
        if np.any(np.isnan(evals)) or np.any(evals < 0):
            return np.zeros(7, dtype=np.float32)
            
        sort_indices = np.argsort(evals)[::-1]
        eval_major = evals[sort_indices[0]]
        eval_minor = evals[sort_indices[1]]

        # 近圆形分布下主轴方向不稳定，固定heading为0可减少框中心/朝向跳变
        anisotropy = eval_major / (eval_minor + 1e-6)
        if anisotropy < 1.1:
            heading = 0.0
        else:
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
    z_min = np.min(points_inlier[:, 2])
    z_max = np.max(points_inlier[:, 2])
    
    # 4. 计算 Box 尺寸 (dx, dy, dz)
    # dx: 沿主轴的长度 (Length)
    # dy: 垂直主轴的宽度 (Width)
    # dz: 高度 (Height)
    l = max(max_xy[0] - min_xy[0], MIN_BOX_SIZE)
    w = max(max_xy[1] - min_xy[1], MIN_BOX_SIZE)
    h = max(z_max - z_min, MIN_BOX_SIZE)
    
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

    box7 = np.array([center_world_xy[0], center_world_xy[1], center_z, l, w, h, heading], dtype=np.float32)

    # 6. 覆盖率校验：确保框能覆盖绝大部分主体点
    # 若覆盖率偏低，按内点统计分位点对尺寸做温和扩张，避免漏包目标点
    coverage = points_inside_box_ratio(points_inlier, box7)
    if coverage < MIN_COVERAGE_RATIO:
        centered_inlier = points_inlier[:, :3] - np.array([box7[0], box7[1], box7[2]], dtype=np.float32)
        local_xy_inlier = np.dot(centered_inlier[:, :2], rotation_matrix.T)

        half_x = np.percentile(np.abs(local_xy_inlier[:, 0]), 99)
        half_y = np.percentile(np.abs(local_xy_inlier[:, 1]), 99)
        half_z = np.percentile(np.abs(centered_inlier[:, 2]), 99)

        box7[3] = max(2.0 * half_x, MIN_BOX_SIZE)
        box7[4] = max(2.0 * half_y, MIN_BOX_SIZE)
        box7[5] = max(2.0 * half_z, MIN_BOX_SIZE)

    return box7

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

        # 注意: 数据集已在源头转换为OpenPCDet坐标系,无需运行时转换
            
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
                if target_points.shape[0] < 70:
                    #print(f"警告: 样本 {sample_idx}.txt 中类别 {class_name} 的点数为 {target_points.shape[0]}（小于30），跳过该目标框生成。")
                    continue

                # 每帧每类只有一个实例，直接计算整体的 Box
                box7 = get_box_from_points(target_points[:, :3], sample_idx=sample_idx, class_name=class_name)

                # 1. 检查标注框体积是否过小
                if box7[3] * box7[4] * box7[5] < 0.2:
                    #print(f"警告: 样本 {sample_idx} 产生极微小体积框，已跳过。")
                    continue
                    
                # 2. 检查是否有 NaN
                if np.any(np.isnan(box7)):
                    #print(f"警告: 样本 {sample_idx} 产生NaN框，已跳过。")
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