import numpy as np
import math
import random
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import DBSCAN

# --- Constants & Configuration ---
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 10

def read_txt(file_path: str) -> Optional[np.ndarray]:
    try:
        data = np.loadtxt(file_path, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"[Error] 读取失败 {file_path}: {e}")
        return None

def save_txt(file_path: str, data: np.ndarray, fmt: str = '%.4f'):
    np.savetxt(file_path, data, fmt=fmt, delimiter=' ')

# --- Geometry Helpers (DRY) ---

def cart2polar(points: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3) [r, theta, phi]"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.clip(r, 1e-6, None)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r)
    return np.stack([r, theta, phi], axis=1)

def polar2cart(r: float, theta: float, phi: float) -> np.ndarray:
    """Scalar polar -> (3,) Cartesian [x, y, z]"""
    cx = r * np.cos(phi) * np.cos(theta)
    cy = r * np.cos(phi) * np.sin(theta)
    cz = r * np.sin(phi)
    return np.array([cx, cy, cz], dtype=np.float32)

def rotate_points_z(points_xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated = points_xyz.copy()
    rotated[:, :2] = np.dot(rotated[:, :2], rot_matrix.T)
    return rotated

# --- Clustering & Grouping Helpers ---

def group_points_by_class(points: np.ndarray) -> List[np.ndarray]:
    """
    [新增/修改] 根据类别标签（Column 4）将点集拆分为实例列表。
    适用于：每个类别在每帧中仅出现一次的情况。
    替代原本对前景物体的 DBSCAN。
    """
    if len(points) == 0:
        return []
    # 获取数据中出现的所有唯一类别标签
    unique_labels = np.unique(points[:, 4].astype(int))
    instances = []
    for lbl in unique_labels:
        # 提取属于该类别的所有点作为一个实例
        cls_points = points[points[:, 4] == lbl]
        instances.append(cls_points)
    return instances

def cluster_points_dbscan(points: np.ndarray, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> List[np.ndarray]:
    """
    保留此函数，用于背景噪声（Bg Noise）的提取，
    因为背景噪声不是单一类别对象，仍需空间聚类。
    """
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    labels = clustering.labels_
    unique_labels = set(labels)
    
    clusters = []
    for lbl in unique_labels:
        if lbl == -1: continue
        clusters.append(points[labels == lbl])
    return clusters

# --- Business Logic ---

def get_fov_bounds(data: np.ndarray) -> dict:
    if data.shape[0] < 10:
        return {'r_range': (0.0, 50.0), 'theta_range': (-np.pi, np.pi), 'phi_range': (-0.5, 0.5)}
    
    polar = cart2polar(data[:, :3])
    r, theta, phi = polar[:, 0], polar[:, 1], polar[:, 2]
    
    r_min, r_max = float(np.percentile(r, 1)), float(np.percentile(r, 100))
    
    theta_range = np.ptp(theta)
    if theta_range > 6.0: 
        theta_min, theta_max = -np.pi, np.pi
    else:
        theta_min, theta_max = float(np.min(theta)), float(np.max(theta))
    
    phi_min, phi_max = float(np.percentile(phi, 1)), float(np.percentile(phi, 99))
    
    return {
        'r_range': (r_min, r_max),
        'theta_range': (theta_min, theta_max),
        'phi_range': (phi_min, phi_max)
    }

def get_water_max_z(data: np.ndarray, default_z: float = 0.0) -> float:
    water_mask = data[:, 4] == 4
    if np.any(water_mask):
        return float(np.max(data[water_mask, 2]))
    return default_z

def get_water_level(data: np.ndarray) -> Optional[float]:
    water_mask = data[:, 4] == 4
    if np.any(water_mask):
        return float(np.max(data[water_mask, 2]))
    return None

def check_object_fully_in_fov(obj_points: np.ndarray, fov_bounds: dict) -> bool:
    if obj_points is None or obj_points.shape[0] == 0: 
        return False
    
    polar = cart2polar(obj_points[:, :3])
    r, theta, phi = polar[:, 0], polar[:, 1], polar[:, 2]
    
    rmin, rmax = fov_bounds['r_range']
    tmin, tmax = fov_bounds['theta_range']
    pmin, pmax = fov_bounds['phi_range']
    
    if np.any((r < rmin) | (r > rmax)): return False
    if np.any((theta < tmin) | (theta > tmax)): return False
    if np.any((phi < pmin) | (phi > pmax)): return False
    
    return True

def check_collision(new_center: np.ndarray, 
                    new_radius: float, 
                    existing_points: np.ndarray, 
                    collision_threshold: float = 0.5) -> bool:
    if existing_points.shape[0] == 0:
        return False
    
    dx = np.abs(existing_points[:, 0] - new_center[0])
    dy = np.abs(existing_points[:, 1] - new_center[1])
    mask = (dx < new_radius + 2.0) & (dy < new_radius + 2.0)
    
    sub_points = existing_points[mask]
    if sub_points.shape[0] == 0:
        return False

    dists = np.linalg.norm(sub_points[:, :3] - new_center, axis=1)
    if np.min(dists) < (new_radius + collision_threshold):
        return True
    return False

def random_transform(points: np.ndarray, 
                     scale_range: Tuple[float, float] = (0.9, 1.1),
                     rotation_range: Tuple[float, float] = (0, 360)) -> np.ndarray:
    points_aug = points.copy()
    scale = np.random.uniform(*scale_range)
    points_aug[:, :3] *= scale
    
    angle = np.random.uniform(*rotation_range)
    points_aug[:, :3] = rotate_points_z(points_aug[:, :3], angle)
    
    return points_aug

def apply_instance_augmentation(points: np.ndarray, probs: dict = None) -> np.ndarray:
    if probs is None:
        probs = {'rotate': 0.5, 'scale': 0.5, 'flip': 0.5, 'shift': 0.5}
        
    points_aug = points.copy()
    center = np.mean(points_aug[:, :3], axis=0)
    xyz_local = points_aug[:, :3] - center
    
    if random.random() < probs['rotate']:
        xyz_local = rotate_points_z(xyz_local, np.random.uniform(0, 360))
        
    if random.random() < probs['scale']:
        xyz_local *= np.random.uniform(0.8, 1.2)

    if random.random() < probs['flip']:
        flip_type = random.choice(['x', 'y', 'xy'])
        if 'x' in flip_type: xyz_local[:, 0] *= -1
        if 'y' in flip_type: xyz_local[:, 1] *= -1

    points_aug[:, :3] = xyz_local + center
    
    if random.random() < probs['shift']:
        shift = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-0.2, 0.2)
        ])
        points_aug[:, :3] += shift
        
    return points_aug

def physically_transform_noise(noise_points: np.ndarray, 
                               target_center_polar: Tuple[float, float, float]) -> np.ndarray:
    if noise_points.shape[0] == 0: return noise_points

    xyz = noise_points[:, :3]
    center_old = np.mean(xyz, axis=0)
    r_old = max(np.linalg.norm(center_old), 1.0)
    
    r_new, t_new, p_new = target_center_polar
    
    scale_factor = np.clip(r_new / r_old, 0.5, 3.0)
    
    xyz_local = xyz - center_old
    xyz_scaled = xyz_local * scale_factor
    
    t_old = np.arctan2(center_old[1], center_old[0])
    delta_t_deg = np.rad2deg(t_new - t_old)
    xyz_scaled = rotate_points_z(xyz_scaled, delta_t_deg)
    
    center_new = polar2cart(r_new, t_new, p_new)
    
    new_points = noise_points.copy()
    new_points[:, :3] = xyz_scaled + center_new
    new_points[:, 4] = 3.0 
    
    return new_points

def physically_transform_object_with_density(obj_points: np.ndarray,
                                             old_center_polar: Tuple[float, float, float],
                                             new_center_polar: Tuple[float, float, float]) -> np.ndarray:
    if obj_points.shape[0] == 0: return obj_points

    r_old, _, _ = old_center_polar
    r_new, t_new, p_new = new_center_polar
    
    ratio = r_old / r_new
    
    xyz_world = obj_points[:, :3]
    center_old_cart = np.mean(xyz_world, axis=0)
    xyz_local = xyz_world - center_old_cart
    xyz_local *= np.random.uniform(0.9, 1.1)
    
    xyz_local = rotate_points_z(xyz_local, np.random.uniform(0, 360))
    
    feats = obj_points[:, 3:]
    num_points = xyz_local.shape[0]
    
    xyz_resampled, feats_resampled = xyz_local, feats
    
    if ratio < 0.95:
        probs = np.random.rand(num_points)
        keep_mask = probs < ratio
        if np.sum(keep_mask) < 5:
            indices = np.random.choice(num_points, min(num_points, 5), replace=False)
            xyz_resampled, feats_resampled = xyz_local[indices], feats[indices]
        else:
            xyz_resampled, feats_resampled = xyz_local[keep_mask], feats[keep_mask]
            
    elif ratio > 1.05:
        target_count = int(num_points * ratio)
        num_new = target_count - num_points
        if num_new > 0:
            indices = np.random.choice(num_points, num_new, replace=True)
            xyz_new = xyz_local[indices] + np.random.normal(0, 0.02, xyz_local[indices].shape)
            xyz_resampled = np.vstack([xyz_local, xyz_new])
            feats_resampled = np.vstack([feats, feats[indices]])

    center_new = polar2cart(r_new, t_new, p_new)
    xyz_final = xyz_resampled + center_new
    
    return np.hstack([xyz_final, feats_resampled])

def apply_gaussian_jitter(points: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    jittered = points.copy()
    noise = np.random.normal(0, sigma, jittered[:, :3].shape)
    jittered[:, :3] += noise
    return jittered

def apply_intensity_based_dropout(points: np.ndarray, drop_rate: float) -> np.ndarray:
    if points.shape[0] < 10: return points
    
    intensities = points[:, 3]
    i_min, i_max = np.min(intensities), np.max(intensities)
    
    n_drop = int(points.shape[0] * drop_rate)
    if n_drop == 0: return points
    
    if i_max - i_min < 1e-6:
        drop_weights = np.ones_like(intensities)
    else:
        drop_weights = (i_max - intensities + 1e-3)
    
    drop_weights /= np.sum(drop_weights)
    
    drop_indices = np.random.choice(points.shape[0], size=n_drop, replace=False, p=drop_weights)
    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[drop_indices] = False
    
    return points[keep_mask]

def simulate_boundary_cutoff(obj_points: np.ndarray, 
                             fov_bounds: dict, 
                             water_max_z: float,
                             min_keep_ratio: float = 0.7,
                             max_keep_ratio: float = 0.95) -> Optional[np.ndarray]:
    center_local = np.mean(obj_points[:, :3], axis=0)
    xyz_local = obj_points[:, :3] - center_local
    radius = np.max(np.linalg.norm(xyz_local, axis=1))
    
    r_rng = fov_bounds['r_range']
    t_rng = fov_bounds['theta_range']
    
    boundary_types = ['r_max', 'theta_edge', 'water_surface']
    random.shuffle(boundary_types)
    
    for b_type in boundary_types:
        new_center = None
        
        if b_type == 'water_surface':
            obj_h = np.max(xyz_local[:, 2]) - np.min(xyz_local[:, 2])
            r = np.random.uniform(r_rng[0]+radius, r_rng[1]-radius)
            t = np.random.uniform(t_rng[0], t_rng[1])
            cx, cy, _ = polar2cart(r, t, 0)
            
            protrusion = np.random.uniform(0.1, 0.4) * obj_h
            z_local_max = np.max(xyz_local[:, 2])
            cz = water_max_z + protrusion - z_local_max
            new_center = np.array([cx, cy, cz])
            
        elif b_type == 'r_max':
            r_target = r_rng[1]
            offset = np.random.uniform(-0.5, 0.5) * radius
            r_c = r_target + offset
            t_c = np.random.uniform(t_rng[0], t_rng[1])
            new_center = polar2cart(r_c, t_c, 0) 
            new_center[2] = np.random.uniform(-2, 2)
            
        elif b_type == 'theta_edge':
            edge = random.choice([t_rng[0], t_rng[1]])
            r_c = np.random.uniform(r_rng[0]+radius, r_rng[1]-radius)
            arc_offset = np.random.uniform(-0.5, 0.5) * radius
            angle_offset = arc_offset / r_c
            t_c = edge + angle_offset
            new_center = polar2cart(r_c, t_c, 0)
            new_center[2] = np.random.uniform(-2, 2)
            
        if new_center is None: continue
        
        cand_points = obj_points.copy()
        cand_points[:, :3] = xyz_local + new_center
        
        valid_mask = cand_points[:, 2] <= water_max_z
        
        polar = cart2polar(cand_points[:, :3])
        r, theta, phi = polar[:, 0], polar[:, 1], polar[:, 2]
        fov_mask = (r >= r_rng[0]) & (r <= r_rng[1]) & \
                   (theta >= t_rng[0]) & (theta <= t_rng[1])
        
        final_mask = valid_mask & fov_mask
        keep_ratio = np.sum(final_mask) / len(final_mask)
        
        if keep_ratio >= min_keep_ratio and keep_ratio < max_keep_ratio:
            return cand_points[final_mask]
            
    return None