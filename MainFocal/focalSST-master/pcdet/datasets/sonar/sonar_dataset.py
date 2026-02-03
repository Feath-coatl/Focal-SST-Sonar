"""
带内存缓存的Sonar数据集类
缓存预处理后的体素化数据(禁用数据增强时使用)

优势:
- 消除体素化CPU瓶颈,直接从内存读取处理好的数据
- 第二个epoch开始速度极快
- 适用于数据集<内存容量且禁用数据增强的场景

使用方法:
1. 在 sonar_dataset.yaml 中设置:
   - CACHE_ALL_DATA: True
   - DISABLE_AUG_LIST: ['random_world_flip', 'random_world_rotation', 'random_world_scaling']
2. 第一个epoch会慢(需预处理),后续epoch极快
"""

import copy
import pickle
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

        # === 1. 强度归一化参数 ===
        self.intensity_clip_max = 4.64e10 
        self.log_norm_divisor = 11.0 

        # === 2. 内存缓存配置 ===
        self.cache_all_data = self.dataset_cfg.get('CACHE_ALL_DATA', False)
        self.preprocessed_cache = {}  # 缓存预处理后的数据
        
        # 3. 加载 Info 文件
        self.sonar_infos = []
        self.include_sonar_data(self.mode)
        
        # === 4. 预加载和预处理数据到内存(训练模式且启用缓存) ===
        if self.cache_all_data and self.training:
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
        返回处理后的点云 [N, 4] (x, y, z, intensity)
        """
        lidar_file = self.root_path / 'points' / f'{idx}.txt'
        assert lidar_file.exists(), f"File not found: {lidar_file}"

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
        如果已缓存预处理数据,直接返回;否则实时处理
        """
        # 如果已缓存预处理数据,直接返回(返回副本避免被修改)
        if index in self.preprocessed_cache:
            return self.preprocessed_cache[index]
        
        # 否则实时处理(验证集或未启用缓存时)
        info = copy.deepcopy(self.sonar_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 读取点云
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

        # 数据增强和预处理
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def __del__(self):
        """清理缓存"""
        if hasattr(self, 'preprocessed_cache'):
            self.preprocessed_cache.clear()
