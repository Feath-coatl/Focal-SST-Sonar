import os
import numpy as np
import pickle
import glob
import argparse
import random
from tqdm import tqdm
import augment_utils as utils
from sklearn.cluster import DBSCAN 

class Config:
    FOREGROUND_CLASSES = [1, 2]
    WATER_CLASS = 4
    BG_CLASS = 3
    
    FORCE_ADD_MAX_TRIALS = 10000 
    
    # Noise Config
    NOISE_MIN = 20
    NOISE_MAX = 50
    NOISE_EPS = 0.8
    NOISE_MIN_SAMPLES = 15
    NOISE_MAX_POINTS = 250

    # Object position limit
    NEW_CENTER_Y = 4.0
    
    # rigid body transform probs
    TRANSFORM_PROBS = {'rotate': 0.5, 'scale': 0.5, 'flip': 0.5, 'shift': 0.5}

class DataAugmentor:
    def __init__(self, db_path):
        print(f"[Init] 加载数据库: {db_path}...")
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.object_db = pickle.load(f)
        else:
            print(f"[Warning] 数据库不存在，跳过库生成: {db_path}")
            self.object_db = {}
            
    def get_existing_objects_and_water(self, data):
        labels = data[:, 4].astype(int)
        fg_mask = np.isin(labels, Config.FOREGROUND_CLASSES)
        return data[fg_mask], data[~fg_mask]

    def extract_real_noise_templates(self, bg_data: np.ndarray) -> list:
        # ... (保持原有逻辑不变)
        candidates = bg_data[bg_data[:, 4] == Config.BG_CLASS]
        if len(candidates) < 50: return []
        sample_indices = np.random.choice(len(candidates), min(len(candidates), 5000), replace=False)
        points_for_cluster = candidates[sample_indices]
        clusters = utils.cluster_points_dbscan(points_for_cluster, 
                                               eps=Config.NOISE_EPS, 
                                               min_samples=Config.NOISE_MIN_SAMPLES)
        noise_templates = [c for c in clusters if len(c) <= Config.NOISE_MAX_POINTS]
        return noise_templates

    def generate_valid_location(self, fov_bounds, obj_points, existing_obstacles, water_max_z, max_trials):
        # ... (保持原有逻辑不变)
        center_local = np.mean(obj_points[:, :3], axis=0)
        xyz_local = obj_points[:, :3] - center_local
        radius = np.max(np.linalg.norm(xyz_local, axis=1))
        obj_height_max = np.max(xyz_local[:, 2])
        r_rng = fov_bounds['r_range']
        t_rng = fov_bounds['theta_range']
        p_rng = fov_bounds['phi_range']
        
        for _ in range(max_trials):
            r_c = random.uniform(r_rng[0] + radius, r_rng[1] - radius)
            t_c = random.uniform(t_rng[0], t_rng[1])
            p_c = random.uniform(p_rng[0], p_rng[1])
            new_center = utils.polar2cart(r_c, t_c, p_c)
            
            if new_center[1] < Config.NEW_CENTER_Y: continue
            if (new_center[2] + obj_height_max) > water_max_z: continue
            
            candidate_points = obj_points.copy()
            candidate_points[:, :3] = xyz_local + new_center
            
            if not utils.check_object_fully_in_fov(candidate_points, fov_bounds): continue
            if utils.check_collision(new_center, radius, existing_obstacles): continue
                
            return new_center, (r_c, t_c, p_c)
        return None, None

    def inject_noise_into_scene(self, scene_objects_list, bg_data, fov_bounds, water_max_z):
        # ... (保持原有逻辑不变)
        noise_templates = self.extract_real_noise_templates(bg_data)
        if not noise_templates: return
        num_noise = random.randint(Config.NOISE_MIN, Config.NOISE_MAX)
        r_min, r_max = fov_bounds['r_range']
        t_min, t_max = fov_bounds['theta_range']
        p_min, p_max = fov_bounds['phi_range']
        for _ in range(num_noise):
            template = random.choice(noise_templates)
            for _try in range(5):
                r_new = random.uniform(max(r_min, 5.0), r_max)
                t_new = random.uniform(t_min, t_max)
                p_new = random.uniform(p_min, p_max)
                noise_blob = utils.physically_transform_noise(template, (r_new, t_new, p_new))
                if noise_blob is None: continue
                if np.max(noise_blob[:, 2]) > water_max_z: continue
                if utils.check_object_fully_in_fov(noise_blob, fov_bounds):
                     scene_objects_list.append(noise_blob)
                     break

    def process_frame(self, data: np.ndarray) -> np.ndarray:
        # Mix 模式：添加新目标到场景中
        fov_bounds = utils.get_fov_bounds(data)
        water_max_z = utils.get_water_max_z(data)
        fg_data, bg_data = self.get_existing_objects_and_water(data)
        final_objects_list = [bg_data]
        scene_obstacles = fg_data[::5, :3] if len(fg_data) > 0 else np.empty((0, 3))
        noise_list = []
        self.inject_noise_into_scene(noise_list, bg_data, fov_bounds, water_max_z)
        final_objects_list.extend(noise_list)
        
        # 原有目标直接保留
        if len(fg_data) > 0:
            final_objects_list.append(fg_data)
        
        existing_classes = set(fg_data[:, 4].astype(int)) if len(fg_data) > 0 else set()
        available_classes = [c for c in self.object_db.keys() 
                             if c not in existing_classes and len(self.object_db[c]) > 0]
        
        # [修改] Mix模式的核心是"混入"新目标，如果有可添加的类但失败了，则返回None
        if available_classes:
            target_cls = random.choice(available_classes)
            
            max_attempts = 50
            new_obj_added = False
            
            for attempt in range(max_attempts):
                db_obj = random.choice(self.object_db[target_cls])
                new_obj_points = db_obj['points'].copy()
                center_local = np.mean(new_obj_points[:, :3], axis=0)
                new_obj_points[:, :3] -= center_local
                new_obj_points = utils.random_transform(new_obj_points)
                
                new_center, _ = self.generate_valid_location(
                    fov_bounds, new_obj_points, scene_obstacles, water_max_z, Config.FORCE_ADD_MAX_TRIALS
                )
                
                if new_center is not None:
                    new_obj_points[:, :3] += new_center
                    
                    # 检查生成的目标点数
                    if new_obj_points.shape[0] >= 100:
                        final_objects_list.append(new_obj_points)
                        new_obj_added = True
                        break
            
            # Mix模式失败：有可添加的类别但50次尝试都失败
            if not new_obj_added:
                return None
        
        return np.vstack(final_objects_list)

    def process_frame_affine(self, data: np.ndarray) -> np.ndarray:
        # Affine 模式：物理变换（位置+密度调整）
        fov_bounds = utils.get_fov_bounds(data)
        water_max_z = utils.get_water_max_z(data)
        fg_data, bg_data = self.get_existing_objects_and_water(data)
        final_objects = [bg_data]
        self.inject_noise_into_scene(final_objects, bg_data, fov_bounds, water_max_z)
        
        has_valid_augmentation = False
        
        if len(fg_data) > 0:
            clusters = utils.group_points_by_class(fg_data)
            processed_obstacles = np.empty((0, 3))
            
            for instance_points in clusters:
                # [修改] 分类处理：≥100点的尝试增强，<100点的保持原样
                if instance_points.shape[0] < 100:
                    # 点数不足，直接保留原样，不尝试变换
                    final_objects.append(instance_points)
                    processed_obstacles = np.vstack([processed_obstacles, instance_points[::5, :3]])
                    continue
                
                center_old_cart = np.mean(instance_points[:, :3], axis=0)
                polar_old = utils.cart2polar(center_old_cart.reshape(1, 3))[0]
                
                # 对≥100点的目标进行变换
                success = False
                for attempt in range(50):
                    new_center_cart, new_center_polar_tuple = self.generate_valid_location(
                        fov_bounds, instance_points, processed_obstacles, water_max_z, max_trials=50
                    )
                    
                    if new_center_cart is not None:
                        transformed = utils.physically_transform_object_with_density(
                            instance_points, polar_old, new_center_polar_tuple
                        )
                        
                        # 检查变换后的点数
                        if transformed.shape[0] >= 100:
                            final_objects.append(transformed)
                            processed_obstacles = np.vstack([processed_obstacles, transformed[::5, :3]])
                            has_valid_augmentation = True
                            success = True
                            break
                    else:
                        # 无法找到有效位置
                        break
                
                if not success:
                    # 变换失败，保留原样
                    final_objects.append(instance_points)
                    processed_obstacles = np.vstack([processed_obstacles, instance_points[::5, :3]])
        
        # [修改] 避免假增强：如果有目标但没有成功增强任何一个，返回None
        # 包括两种情况：1) 所有目标<100点  2) 有≥100点的目标但都变换失败
        if len(fg_data) > 0 and not has_valid_augmentation:
            return None
        
        return np.vstack(final_objects)

    def process_frame_cutoff(self, data: np.ndarray) -> np.ndarray:
        """
        Cutoff Mode:
        1. Keep background.
        2. Inject noise.
        3. For existing objects:
           - Size < 150 -> Keep original (cutoff would result in < 105 points).
           - Size >= 150 -> Try cutoff (preserve 70-95%, result >= 100 points).
        4. Return None only if NO objects were successfully augmented.
        """
        fov_bounds = utils.get_fov_bounds(data)
        water_max_z = utils.get_water_max_z(data)
        fg_data, bg_data = self.get_existing_objects_and_water(data)
        
        final_objects = [bg_data]
        self.inject_noise_into_scene(final_objects, bg_data, fov_bounds, water_max_z)
        
        has_valid_augmentation = False
        
        if len(fg_data) > 0:
            clusters = utils.group_points_by_class(fg_data)
            for instance_points in clusters:
                # [修改] 智能预判：cutoff保留70-95%，需要原始≥150点才能保证结果≥105点
                # 150 * 0.7 = 105 (安全边界)
                if instance_points.shape[0] < 150:
                    # 点数不足，cutoff后必然<105，直接保留原样
                    final_objects.append(instance_points)
                    continue
                
                # 对≥150点的目标尝试cutoff
                success = False
                for _ in range(50):
                    # 随机变换增加多样性
                    aug_points = utils.random_transform(instance_points, 
                                                      scale_range=(0.9, 1.1), 
                                                      rotation_range=(0, 360))
                    
                    # 尝试边界截断
                    cutoff_points = utils.simulate_boundary_cutoff(aug_points, fov_bounds, water_max_z)
                    
                    # 检查cutoff后的点数
                    if cutoff_points is not None and cutoff_points.shape[0] >= 100:
                        final_objects.append(cutoff_points)
                        has_valid_augmentation = True
                        success = True
                        break
                
                if not success:
                    # Cutoff失败，保留原样
                    final_objects.append(instance_points)
                    
        # [修改] 避免假增强：如果有目标但没有成功cutoff任何一个，返回None
        # 包括两种情况：1) 所有目标<150点  2) 有≥150点的目标但都cutoff失败
        if len(fg_data) > 0 and not has_valid_augmentation:
            return None
                    
        return np.vstack(final_objects)

    def process_frame_dropout_jitter(self, data: np.ndarray) -> np.ndarray:
        """
        Dropout & Jitter Mode:
        - In-place rotation/scaling.
        - Intensity-based dropout (5%~15%).
        - Gaussian Jitter.
        """
        fov_bounds = utils.get_fov_bounds(data)
        water_max_z = utils.get_water_max_z(data)
        fg_data, bg_data = self.get_existing_objects_and_water(data)
        
        final_objects = [bg_data]
        self.inject_noise_into_scene(final_objects, bg_data, fov_bounds, water_max_z)
        
        has_valid_augmentation = False
        
        if len(fg_data) > 0:
            clusters = utils.group_points_by_class(fg_data)
            for instance_points in clusters:
                # [修改] 智能预判：dropout最多15%，需要原始≥118点才能保证结果≥100点
                # 118 * 0.85 = 100.3 (安全边界)
                if instance_points.shape[0] < 118:
                    # 点数不足，dropout后必然<100，直接保留原样
                    final_objects.append(instance_points)
                    continue
                
                center = np.mean(instance_points[:, :3], axis=0)
                xyz_local = instance_points[:, :3] - center
                
                # 对≥118点的目标尝试dropout
                success = False
                for attempt in range(50):
                    aug_points = instance_points.copy()
                    aug_points[:, :3] = xyz_local
                    
                    aug_points = utils.random_transform(aug_points, 
                                                        scale_range=(0.9, 1.1),
                                                        rotation_range=(0, 360))
                    
                    drop_rate = np.random.uniform(0.05, 0.15)
                    aug_points = utils.apply_intensity_based_dropout(aug_points, drop_rate)
                    
                    # 检查dropout后的点数
                    if aug_points.shape[0] >= 100:
                        aug_points = utils.apply_gaussian_jitter(aug_points, sigma=0.02)
                        aug_points[:, :3] += center
                        final_objects.append(aug_points)
                        has_valid_augmentation = True
                        success = True
                        break
                
                if not success:
                    # Dropout失败，保留原样
                    final_objects.append(instance_points)
        
        # [修改] 避免假增强：如果有目标但没有成功dropout任何一个，返回None
        # 包括两种情况：1) 所有目标<118点  2) 有≥118点的目标但都dropout失败
        if len(fg_data) > 0 and not has_valid_augmentation:
            return None
                
        return np.vstack(final_objects)

def save_result(data_aug, original_path, src_root, dst_root, prefix):
    rel_path = os.path.relpath(original_path, src_root)
    dirname, filename = os.path.split(rel_path)
    new_filename = f"{prefix}{filename}"
    new_rel_path = os.path.join(dirname, new_filename)
    save_path = os.path.join(dst_root, new_rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    utils.save_txt(save_path, data_aug)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--db', type=str, default='object_db.pkl')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'affine', 'mix', 'cutoff', 'dropout'],
                        help="Choose augmentation mode.")
    args = parser.parse_args()
    
    augmentor = DataAugmentor(args.db)
    
    all_files = sorted(glob.glob(os.path.join(args.src, "**/*.txt"), recursive=True))
    total_files = len(all_files)
    print(f"[Info] 源目录共找到 {total_files} 个文件。 Mode: {args.mode}")

    if total_files == 0:
        return

    # 1. Affine Mode
    if args.mode in ['all', 'affine']:
        print(f"\n[Mode: Affine] 正在执行 (100% 数据, 前缀 a_)...")
        affine_success = 0
        affine_failed = 0
        for fpath in tqdm(all_files, desc="Affine"):
            data = utils.read_txt(fpath)
            if data is None: continue
            data_aug = augmentor.process_frame_affine(data)
            if data_aug is not None:
                save_result(data_aug, fpath, args.src, args.dst, "a_")
                affine_success += 1
            else:
                affine_failed += 1
        print(f"  [Info] Affine 模式: 成功 {affine_success}, 失败 {affine_failed}")

    # 2. Mix Mode
    if args.mode in ['all', 'mix']:
        sample_size = int(total_files * 0.4)
        mix_files = random.sample(all_files, sample_size) 
        print(f"\n[Mode: Mix] 正在执行 (采样 {len(mix_files)} 个文件, 前缀 m_)...")
        mix_success = 0
        mix_failed = 0
        for fpath in tqdm(mix_files, desc="Mix"):
            data = utils.read_txt(fpath)
            if data is None: continue
            data_aug = augmentor.process_frame(data)
            if data_aug is not None:
                save_result(data_aug, fpath, args.src, args.dst, "m_")
                mix_success += 1
            else:
                mix_failed += 1
        print(f"  [Info] Mix 模式: 成功 {mix_success}, 失败 {mix_failed}")

    # 3. Cutoff Mode
    if args.mode in ['all', 'cutoff']:
        sample_size = int(total_files * 0.4)
        mix_files = random.sample(all_files, sample_size) 
        print(f"\n[Mode: Cutoff] 正在执行 (采样 {len(mix_files)} 个文件, 前缀 c_)...")
        count_saved = 0
        for fpath in tqdm(mix_files, desc="Cutoff"):
            data = utils.read_txt(fpath)
            if data is None: continue
            
            # process_frame_cutoff now returns None if no changes occurred
            data_aug = augmentor.process_frame_cutoff(data)
            
            if data_aug is not None:
                save_result(data_aug, fpath, args.src, args.dst, "c_")
                count_saved += 1
        print(f"  [Info] Cutoff 模式实际生成文件数: {count_saved} / {sample_size}")

    # 4. Dropout & Jitter Mode
    if args.mode in ['all', 'dropout']:
        print(f"\n[Mode: Dropout] 正在执行 (100% 数据, 前缀 d_)...")
        dropout_success = 0
        dropout_failed = 0
        for fpath in tqdm(all_files, desc="Dropout"):
            data = utils.read_txt(fpath)
            if data is None: continue
            data_aug = augmentor.process_frame_dropout_jitter(data)
            if data_aug is not None:
                save_result(data_aug, fpath, args.src, args.dst, "d_")
                dropout_success += 1
            else:
                dropout_failed += 1
        print(f"  [Info] Dropout 模式: 成功 {dropout_success}, 失败 {dropout_failed}")

    print("\n[Success] 全部任务完成。")

if __name__ == "__main__":
    main()