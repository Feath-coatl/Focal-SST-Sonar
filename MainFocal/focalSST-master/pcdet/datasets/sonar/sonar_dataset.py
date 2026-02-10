"""
带内存缓存的Sonar数据集类
支持两种缓存模式:
1. 运行时缓存(CACHE_ALL_DATA): 第一个epoch预处理并缓存,后续epoch快速
2. 磁盘缓存(USE_DISK_CACHE): 从预先生成的缓存文件加载,所有epoch都快

使用方法:
方式1 - 运行时缓存:
  - CACHE_ALL_DATA: True
  - DISABLE_AUG_LIST: ['random_world_flip', 'random_world_rotation', 'random_world_scaling']

方式2 - 磁盘缓存(推荐):
  - 先运行: python tools/preprocess_dataset.py --cfg_file xxx.yaml
  - USE_DISK_CACHE: True
  - DISABLE_AUG_LIST: ['random_world_flip', 'random_world_rotation', 'random_world_scaling']
"""

import copy
import pickle
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..dataset import DatasetTemplate

class SonarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path: data/sonar
            dataset_cfg: 配置文件
            class_names: 类别名称
            training: 训练模式标志
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 根据 yaml 中的配置读取 split (train 或 val)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        
        # 类名映射（用于评估）
        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI', None)

        # === 1. 强度归一化参数 ===
        self.intensity_clip_max = 4.71e10 #强度最大值在1.96e11 99.9%分位点强度值为4.71e10
        self.log_norm_divisor = 11.0 

        # === 2. 内存缓存配置 ===
        self.cache_all_data = self.dataset_cfg.get('CACHE_ALL_DATA', False)
        self.use_disk_cache = self.dataset_cfg.get('USE_DISK_CACHE', False)
        self.use_binary_format = self.dataset_cfg.get('USE_BINARY_FORMAT', False)
        self.preprocessed_cache = {}  # 缓存预处理后的数据
        
        # 3. 加载 Info 文件
        self.sonar_infos = []
        self.include_sonar_data(self.mode)
        
        # === 4. 加载缓存数据 ===
        if self.use_disk_cache and self.training:
            # 从磁盘加载预处理缓存(最快)
            self.load_disk_cache()
        elif self.cache_all_data and self.training:
            # 运行时预处理并缓存
            self.preload_and_preprocess_all_data()

    def include_sonar_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Sonar dataset...')

        self.sonar_infos = []
        
        if self.dataset_cfg.get('INFO_PATH', None) is None:
            info_path = self.root_path / f'sonar_infos_{self.split}.pkl'
            infos = []
            if info_path.exists():
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
            self.sonar_infos.extend(infos)
        else:
            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    self.sonar_infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total samples for sonar dataset: %d' % (len(self.sonar_infos)))

    def load_disk_cache(self):
        """
        从磁盘加载预处理缓存文件
        这是最快的方式 - 连第一个epoch都很快
        """
        # 计算配置哈希
        config_str = str(self.dataset_cfg.DATA_PROCESSOR)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_dir = self.root_path / 'preprocessed_cache' / config_hash
        
        if not cache_dir.exists() or not (cache_dir / 'COMPLETED').exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"❌ 预处理缓存不存在或未完成\n"
                f"缓存目录: {cache_dir}\n"
                f"配置哈希: {config_hash}\n"
                f"\n请先运行以下命令生成缓存:\n"
                f"  cd tools\n"
                f"  python preprocess_dataset.py --cfg_file cfgs/sonar_models/focal_sst.yaml\n"
                f"{'='*60}\n"
            )
        
        if self.logger is not None:
            self.logger.info(f'Loading preprocessed cache from disk...')
            self.logger.info(f'Cache dir: {cache_dir}')
            self.logger.info(f'Config hash: {config_hash}')
        
        # 加载所有缓存文件
        success_count = 0
        for info in tqdm(self.sonar_infos, desc="Loading cache"):
            sample_idx = info['point_cloud']['lidar_idx']
            cache_file = cache_dir / f'{sample_idx}.pkl'
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.preprocessed_cache[sample_idx] = pickle.load(f)
                success_count += 1
            else:
                if self.logger is not None:
                    self.logger.warning(f'Cache file missing: {cache_file}')
        
        if self.logger is not None:
            self.logger.info(f'✅ Loaded {success_count}/{len(self.sonar_infos)} cached samples')
            self.logger.info('⚡ All epochs will be FAST!')

    def preload_and_preprocess_all_data(self):
        """
        预加载所有数据并完成预处理(体素化等)
        警告: 需要大量内存 (约数据集大小的3-5倍)
        """
        if self.logger is not None:
            self.logger.info(f'Preloading and preprocessing {len(self.sonar_infos)} samples...')
            self.logger.info('This will take several minutes but dramatically speed up training!')
        
        for index in tqdm(range(len(self.sonar_infos)), desc="Preprocessing data"):
            # 使用原始数据处理流程
            info = copy.deepcopy(self.sonar_infos[index])
            sample_idx = info['point_cloud']['lidar_idx']
            
            # 加载点云
            points = self._load_points_from_file(sample_idx)
            
            input_dict = {
                'points': points,
                'frame_id': sample_idx,
            }

            # 加载GT Boxes
            if 'annos' in info:
                annos = info['annos']
                gt_boxes = annos.get('gt_boxes_lidar', annos.get('gt_boxes', None))
                gt_names = annos.get('name', annos.get('gt_names', None))

                if isinstance(gt_boxes, list): 
                    gt_boxes = np.array(gt_boxes)
                if isinstance(gt_names, list): 
                    gt_names = np.array(gt_names)

                if gt_boxes is not None and gt_names is not None and len(gt_boxes) > 0:
                    if self.class_names is not None:
                        mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)
                        gt_boxes = gt_boxes[mask]
                        gt_names = gt_names[mask]

                    input_dict.update({
                        'gt_boxes': gt_boxes,
                        'gt_names': gt_names
                    })

            # 执行预处理(体素化等) - 这是最耗时的部分
            data_dict = self.prepare_data(data_dict=input_dict)
            
            # 缓存处理好的数据
            self.preprocessed_cache[index] = data_dict
        
        if self.logger is not None:
            self.logger.info(f'Successfully cached {len(self.preprocessed_cache)} preprocessed samples')
            self.logger.info('Subsequent epochs will be MUCH faster!')

    def _load_points_from_file(self, idx):
        """
        从文件加载原始点云数据
        支持两种格式:
        - 文本格式(.txt): np.loadtxt() - 慢但可读
        - 二进制格式(.bin): np.fromfile() - 快5-10倍
        
        返回处理后的点云 [N, 4] (x, y, z, intensity)
        """
        if self.use_binary_format:
            # 二进制格式: 速度快, 空间小
            lidar_file = self.root_path / 'points_binary' / f'{idx}.bin'
            assert lidar_file.exists(), f"Binary file not found: {lidar_file}"
            
            # 直接从二进制文件加载 (float32, 5列)
            points_all = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
        else:
            # 文本格式: 兼容性好, 可读
            lidar_file = self.root_path / 'points' / f'{idx}.txt'
            assert lidar_file.exists(), f"Text file not found: {lidar_file}"
            
            # 加载数据 [x, y, z, intensity, class]
            points_all = np.loadtxt(str(lidar_file), dtype=np.float32)
            if points_all.ndim == 1:
                points_all = points_all.reshape(1, -1)

        # 提取 x, y, z, intensity (前4列)
        points = points_all[:, :4]

        # === 强度归一化逻辑 ===
        intensity = points[:, 3]
        # 截断离群值 (Clip outliers > p99.9)
        intensity = np.clip(intensity, a_min=0, a_max=self.intensity_clip_max)
        # 方案A. 对数变换 (Log Transform)
        intensity = np.log10(intensity + 1)
        intensity = intensity / self.log_norm_divisor
        # 方案B. 线性缩放 (Scale to 0~1)
        #intensity = np.clip(intensity / self.intensity_clip_max, 0, 1)
        points[:, 3] = intensity

        # 数据已在源头转换为OpenPCDet坐标系,直接返回
        return points

    def __len__(self):
        return len(self.sonar_infos)

    def __getitem__(self, index):
        """
        获取数据
        优先级: 磁盘缓存 > 内存缓存 > 实时处理
        """
        info = self.sonar_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 1. 尝试从磁盘缓存读取
        if self.use_disk_cache and sample_idx in self.preprocessed_cache:
            return self.preprocessed_cache[sample_idx]
        
        # 2. 尝试从内存缓存读取
        if index in self.preprocessed_cache:
            return self.preprocessed_cache[index]
        
        # 3. 实时处理(验证集或未启用缓存)
        info = copy.deepcopy(info)
        points = self._load_points_from_file(sample_idx)
        
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        # 加载GT Boxes
        if 'annos' in info:
            annos = info['annos']
            gt_boxes = annos.get('gt_boxes_lidar', annos.get('gt_boxes', None))
            gt_names = annos.get('name', annos.get('gt_names', None))

            if isinstance(gt_boxes, list): 
                gt_boxes = np.array(gt_boxes)
            if isinstance(gt_names, list): 
                gt_names = np.array(gt_names)

            if gt_boxes is not None and gt_names is not None and len(gt_boxes) > 0:
                if self.class_names is not None:
                    mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)
                    gt_boxes = gt_boxes[mask]
                    gt_names = gt_names[mask]

                if len(gt_boxes) > 0:
                    input_dict.update({
                        'gt_boxes': gt_boxes,
                        'gt_names': gt_names
                    })
        
        # 训练时如果没有gt_boxes，提供空数组而不是跳过
        # 这样可以让模型学习背景
        if self.training and 'gt_boxes' not in input_dict:
            input_dict.update({
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                'gt_names': np.array([], dtype='<U10')
            })

        # 数据增强和预处理
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        """
        评估检测结果
        使用KITTI评估标准计算AP指标
        
        Args:
            det_annos: 检测结果列表
            class_names: 类别名称列表 (e.g., ['Box', 'Diver'])
            **kwargs: 其他参数，包括 eval_metric
        
        Returns:
            result_str: 评估结果字符串
            result_dict: 评估指标字典
        """
        if 'annos' not in self.sonar_infos[0].keys():
            return 'No ground-truth annotations for evaluation', {}
        
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.sonar_infos]
        
        # 将声纳类名转换为KITTI标准类名
        if self.map_class_to_kitti is not None:
            kitti_utils.transform_annotations_to_kitti_format(
                eval_det_annos, 
                map_name_to_kitti=self.map_class_to_kitti
            )
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, 
                map_name_to_kitti=self.map_class_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            # 转换类名列表
            kitti_class_names = [self.map_class_to_kitti[x] for x in class_names]
        else:
            kitti_class_names = class_names
        
        # 使用KITTI评估方法
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, 
            dt_annos=eval_det_annos, 
            current_classes=kitti_class_names
        )
        
        return ap_result_str, ap_dict

    def __del__(self):
        """清理缓存"""
        if hasattr(self, 'preprocessed_cache'):
            self.preprocessed_cache.clear()
