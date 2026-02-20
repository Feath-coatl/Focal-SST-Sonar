"""
为Sonar数据集创建GT数据库 (Ground Truth Database)

GT数据库采样是3D目标检测中最重要的数据增强技术之一。
它将训练集中每个GT目标的点云单独保存，训练时随机"粘贴"到当前场景中，
大幅增加少数类别（如Diver）的训练样本数量。

使用方法:
    cd focalSST-master
    python create_sonar_gt_database.py

输出:
    data/sonar/gt_database/             - 每个GT目标的点云文件(.bin)
    data/sonar/sonar_dbinfos_train.pkl   - GT数据库信息索引
"""

import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
ROOT_PATH = Path('data/sonar')
POINTS_DIR = ROOT_PATH / 'points'
POINTS_BINARY_DIR = ROOT_PATH / 'points_binary'
USE_BINARY = True  # 使用二进制格式（更快）

# 类别映射
CLASS_MAPPING = {1: 'Box', 2: 'Diver'}

# 强度归一化参数（必须与sonar_dataset.py一致）
INTENSITY_CLIP_MAX = 4.71e10
LOG_NORM_DIVISOR = 11.0

# 最少点数阈值（与create_sonar_infos.py一致）
MIN_POINTS_THRESHOLD = 70

# 去噪参数（与create_sonar_infos.py保持一致）
MIN_INLIER_POINTS = 20
ROBUST_ZSCORE_THRESHOLD = 3.5
MIN_INLIER_RATIO = 0.55
MAX_REMOVAL_RATIO = 0.05
# ===========================================


def normalize_intensity(intensity):
    """强度归一化，与sonar_dataset.py的_load_points_from_file完全一致"""
    intensity = np.clip(intensity, a_min=0, a_max=INTENSITY_CLIP_MAX)
    intensity = np.log10(intensity + 1)
    intensity = intensity / LOG_NORM_DIVISOR
    return intensity


def load_points(sample_idx):
    """加载原始点云（5列: x, y, z, intensity, class）"""
    if USE_BINARY and (POINTS_BINARY_DIR / f'{sample_idx}.bin').exists():
        points_all = np.fromfile(
            str(POINTS_BINARY_DIR / f'{sample_idx}.bin'), 
            dtype=np.float32
        ).reshape(-1, 5)
    else:
        txt_path = POINTS_DIR / f'{sample_idx}.txt'
        if not txt_path.exists():
            return None
        points_all = np.loadtxt(str(txt_path), dtype=np.float32)
        if points_all.ndim == 1:
            points_all = points_all.reshape(1, -1)
    return points_all


def robust_filter_object_points(points_xyz, sample_idx=None, class_name=None):
    """
    对单目标点做鲁棒去噪，并硬约束最多剔除5%。
    返回过滤后的点索引掩码与剔除点数。
    """
    n_points = points_xyz.shape[0]
    if n_points < MIN_INLIER_POINTS:
        return np.ones(n_points, dtype=bool), 0

    median_xyz = np.median(points_xyz, axis=0)
    mad_xyz = np.median(np.abs(points_xyz - median_xyz), axis=0)
    robust_scale = 1.4826 * mad_xyz + 1e-3
    robust_dist = np.sqrt(np.sum(((points_xyz - median_xyz) / robust_scale) ** 2, axis=1))
    mask_mad = robust_dist <= ROBUST_ZSCORE_THRESHOLD

    center_xy = np.median(points_xyz[:, :2], axis=0)
    radial_dist = np.linalg.norm(points_xyz[:, :2] - center_xy, axis=1)
    q1, q3 = np.percentile(radial_dist, [25, 75])
    iqr = max(q3 - q1, 1e-3)
    radial_thresh = q3 + 1.5 * iqr
    mask_radial = radial_dist <= radial_thresh

    mask = mask_mad & mask_radial

    max_remove = int(np.floor(n_points * MAX_REMOVAL_RATIO))
    min_keep_by_ratio = n_points - max_remove
    min_keep = max(MIN_INLIER_POINTS, int(n_points * MIN_INLIER_RATIO), min_keep_by_ratio)

    if np.sum(mask) < min_keep:
        # 过滤过激时，回退为仅删除异常分数最高的最多5%点
        if max_remove <= 0:
            keep_mask = np.ones(n_points, dtype=bool)
        else:
            radial_norm = radial_dist / (np.median(radial_dist) + 1e-3)
            anomaly_score = robust_dist + radial_norm
            keep_indices = np.argsort(anomaly_score)[:min_keep_by_ratio]
            keep_mask = np.zeros(n_points, dtype=bool)
            keep_mask[keep_indices] = True
    else:
        keep_mask = mask

    removed_count = int(n_points - np.sum(keep_mask))
    if removed_count > 0:
        removed_ratio = removed_count / n_points
        if sample_idx is not None and class_name is not None:
            print(f"[GT去噪] 样本 {sample_idx} 类别 {class_name}: 剔除 {removed_count}/{n_points} ({removed_ratio:.2%})")
        else:
            print(f"[GT去噪] 剔除 {removed_count}/{n_points} ({removed_ratio:.2%})")

    return keep_mask, removed_count


def create_gt_database():
    """创建GT数据库"""
    
    # 1. 加载训练集信息
    info_path = ROOT_PATH / 'sonar_infos_train.pkl'
    if not info_path.exists():
        print(f"错误: 找不到 {info_path}")
        print("请先运行 python create_sonar_infos.py 生成训练集信息")
        return
    
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"加载了 {len(infos)} 个训练样本")
    
    # 2. 创建输出目录
    db_save_path = ROOT_PATH / 'gt_database'
    db_save_path.mkdir(parents=True, exist_ok=True)
    
    all_db_infos = {}
    
    # 统计
    stats = {name: {'count': 0, 'total_points': 0, 'min_points': float('inf'), 'max_points': 0}
             for name in CLASS_MAPPING.values()}
    
    # 3. 遍历每个训练样本
    for k in tqdm(range(len(infos)), desc="Creating GT database"):
        info = infos[k]
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 检查是否有标注
        if 'annos' not in info:
            continue
        
        annos = info['annos']
        gt_names = annos.get('name', None)
        gt_boxes = annos.get('gt_boxes_lidar', None)
        
        if gt_names is None or gt_boxes is None or len(gt_boxes) == 0:
            continue
        
        if isinstance(gt_names, np.ndarray) and gt_names.size == 0:
            continue
        
        # 加载原始点云（包含类别列）
        points_all = load_points(sample_idx)
        if points_all is None:
            continue
        
        # 处理强度（与sonar_dataset.py一致）
        points_processed = points_all[:, :4].copy()
        points_processed[:, 3] = normalize_intensity(points_processed[:, 3])
        class_labels = points_all[:, 4].astype(int)
        
        # 4. 对每个GT目标提取点云
        for i in range(len(gt_boxes)):
            name = gt_names[i] if isinstance(gt_names[i], str) else str(gt_names[i])
            box = gt_boxes[i]  # [x, y, z, dx, dy, dz, heading]
            
            # 确定类别对应的class_id
            class_id = None
            for cid, cname in CLASS_MAPPING.items():
                if cname == name:
                    class_id = cid
                    break
            
            if class_id is None:
                continue
            
            # 通过类别标签提取该目标的点（每帧每类只有一个实例）
            mask = (class_labels == class_id)
            gt_points = points_processed[mask].copy()
            
            if gt_points.shape[0] < 5:  # 太少的点跳过
                continue

            # 与create_sonar_infos.py一致：先做鲁棒去噪，且剔除比例不超过5%
            keep_mask, _ = robust_filter_object_points(
                gt_points[:, :3],
                sample_idx=sample_idx,
                class_name=name
            )
            gt_points = gt_points[keep_mask]

            if gt_points.shape[0] < 5:
                continue
            
            # 将点平移到以GT box中心为原点
            gt_points[:, 0] -= box[0]
            gt_points[:, 1] -= box[1]
            gt_points[:, 2] -= box[2]
            
            # 保存为.bin文件
            filename = f'{sample_idx}_{name}_{i}.bin'
            filepath = db_save_path / filename
            gt_points.astype(np.float32).tofile(str(filepath))
            
            # 记录db_info
            db_info = {
                'name': name,
                'path': str(Path('gt_database') / filename),  # 相对路径
                'image_idx': sample_idx,
                'gt_idx': i,
                'box3d_lidar': box.astype(np.float32),
                'num_points_in_gt': gt_points.shape[0],
                'difficulty': 0,  # 可以根据点数或大小设置难度
            }
            
            if name not in all_db_infos:
                all_db_infos[name] = []
            all_db_infos[name].append(db_info)
            
            # 更新统计
            stats[name]['count'] += 1
            stats[name]['total_points'] += gt_points.shape[0]
            stats[name]['min_points'] = min(stats[name]['min_points'], gt_points.shape[0])
            stats[name]['max_points'] = max(stats[name]['max_points'], gt_points.shape[0])
    
    # 5. 保存数据库信息
    db_info_path = ROOT_PATH / 'sonar_dbinfos_train.pkl'
    with open(db_info_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
    
    # 6. 打印统计信息
    print("\n" + "=" * 60)
    print("GT Database 创建完成!")
    print("=" * 60)
    print(f"数据库路径: {db_save_path}")
    print(f"信息文件:   {db_info_path}")
    print()
    
    for name, stat in stats.items():
        if stat['count'] > 0:
            avg = stat['total_points'] / stat['count']
            print(f"  {name}:")
            print(f"    数量: {stat['count']}")
            print(f"    点数: min={stat['min_points']}, max={stat['max_points']}, avg={avg:.0f}")
        else:
            print(f"  {name}: 无有效样本")
    
    print()
    print("接下来请在YAML配置中启用gt_sampling增强:")
    print("  DATA_AUGMENTOR:")
    print("    AUG_CONFIG_LIST:")
    print("      - NAME: gt_sampling")
    print("        DB_INFO_PATH: ['sonar_dbinfos_train.pkl']")
    print("        SAMPLE_GROUPS: ['Box:3', 'Diver:5']")


if __name__ == '__main__':
    create_gt_database()
