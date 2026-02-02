import copy
import pickle
import numpy as np
from pathlib import Path

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

        # === 1. 强度归一化参数 (基于你的分析报告) ===
        # 数据极大值约 1.95e11，p99.9 约 4.64e10
        # 策略：Log(x + 1) / Norm_Factor
        # Log10(4.64e10) ≈ 10.66
        self.intensity_clip_max = 4.64e10 
        self.log_norm_divisor = 11.0 

        # 2. 加载 Info 文件 (关键修改)
        self.sonar_infos = []
        self.include_sonar_data(self.mode)

    def include_sonar_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Sonar dataset...')

        self.sonar_infos = []
        
        # 确保 yaml 中定义了 INFO_PATH
        # 如果 yaml 没定义，默认寻找 standard 路径
        if self.dataset_cfg.get('INFO_PATH', None) is None:
            # 默认回退方案
            info_path = self.root_path / f'sonar_infos_{self.split}.pkl'
            infos = []
            if info_path.exists():
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
            self.sonar_infos.extend(infos)
        else:
            # 标准方案：从配置文件读取路径列表
            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    self.sonar_infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total samples for sonar dataset: %d' % (len(self.sonar_infos)))

    def get_lidar(self, idx):
        """
        核心函数：根据索引读取具体文件
        """
        # 修改点：指向 points 子文件夹
        lidar_file = self.root_path / 'points' / f'{idx}.txt'
        assert lidar_file.exists(), f"File not found: {lidar_file}"

        # 加载数据 [x, y, z, intensity, class]
        points_all = np.loadtxt(str(lidar_file), dtype=np.float32)
        # 确保维度正确 (防止某些文件少于5列报错)
        if points_all.ndim == 1:
            points_all = points_all.reshape(1, -1)

        # 提取 x, y, z, intensity (前4列输入网络)
        points = points_all[:, :4]
        # (可选) 可以在这里保留第5列 class 用于后续的分割辅助任务
        # labels = points_all[:, 4] 

        # === 3. 强度归一化逻辑 ===
        intensity = points[:, 3]
        # 截断离群值 (Clip outliers > p99.9)
        intensity = np.clip(intensity, a_min=0, a_max=self.intensity_clip_max)
        # 方案A. 对数变换 (Log Transform)
        intensity = np.log10(intensity + 1)
        intensity = intensity / self.log_norm_divisor
        # 方案B. 线性缩放 (Scale to 0~1)
        #intensity = np.clip(intensity / self.intensity_clip_max, 0, 1)
        points[:, 3] = intensity

        # 坐标系对齐 (Coordinate Alignment)
        # 我的数据: X右, Y前, Z上
        # PCDet标准: X前, Y左, Z上
        points_aligned = np.zeros_like(points)
        points_aligned[:, 0] = points[:, 1]  # Old Y -> New X
        points_aligned[:, 1] = -points[:, 0] # -Old X -> New Y
        points_aligned[:, 2] = points[:, 2]  # Z 保留
        points_aligned[:, 3] = points[:, 3]  # Intensity 保留

        return points_aligned

    def __len__(self):
        return len(self.sonar_infos)

    def __getitem__(self, index):
        info = copy.deepcopy(self.sonar_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # === 打印当前正在加载的帧 ID ===
        # print(f"DEBUG: Loading frame {sample_idx}", flush=True)
        # 读取点云
        points = self.get_lidar(sample_idx)
        
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        # === 关键修正：正确加载 GT Boxes 和 GT Names ===
        if 'annos' in info:
            annos = info['annos']
            # 从 infos 中提取 gt_boxes_lidar 和 name
            # 你的 create_sonar_infos.py 生成的键是 'gt_boxes_lidar' 和 'name'
            gt_boxes = annos.get('gt_boxes_lidar', annos.get('gt_boxes', None))
            gt_names = annos.get('name', annos.get('gt_names', None))

            # 只有当两者都存在且不为空时才处理
            if gt_boxes is not None and gt_names is not None and len(gt_boxes) > 0:
                # 使用 self.class_names 进行过滤 (这是从配置文件 CLASS_NAMES 传进来的)
                if self.class_names is not None:
                    mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)
                    gt_boxes = gt_boxes[mask]
                    gt_names = gt_names[mask]

                # 无论是否过滤，只要有 gt_boxes 就必须同时更新 gt_names
                input_dict.update({
                    'gt_boxes': gt_boxes,
                    'gt_names': gt_names
                })

        # 数据增强 (训练时开启)
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict