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
        
        # 1. 加载我们在第一步生成的 txt 列表
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_file_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []

        self.sonar_infos = []
        self.include_sonar_data(self.mode)

        # === 2. 强度归一化参数 (基于你的分析报告) ===
        # 你的数据极大值约 1.95e11，p99.9 约 4.64e10
        # 策略：Log(x + 1) / Norm_Factor
        # Log10(4.64e10) ≈ 10.66，我们取 12.0 作为分母，将数据映射到 [0, 1] 附近
        self.intensity_clip_max = 4.64e10 
        self.log_norm_divisor = 11.0 

    def include_sonar_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Sonar dataset...')

        # 将文件名列表转换为 info 字典列表
        for sample_idx in self.sample_file_list:
            info = {'point_cloud': {'lidar_idx': sample_idx}}
            # 暂时还没有生成的 .pkl 标注文件，这里仅构建基础索引
            # 后续生成 infos.pkl 后，这里会改为从 pkl 加载
            self.sonar_infos.append(info)

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

        # 1. 提取 x, y, z, intensity (前4列输入网络)
        points = points_all[:, :4]
        
        # (可选) 你可以在这里保留第5列 class 用于后续的分割辅助任务
        # labels = points_all[:, 4] 

        # === 3. 强度归一化逻辑 ===
        intensity = points[:, 3]
        
        # 截断离群值 (Clip outliers > p99.9)
        intensity = np.clip(intensity, a_min=0, a_max=self.intensity_clip_max)
        
        # 方案A. 对数变换 (Log Transform)
        intensity = np.log1p(intensity)
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
        
        # 读取点云
        points = self.get_lidar(sample_idx)
        
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        # 暂时跳过 GT 标注加载 (因为还没生成 infos.pkl)
        # 仅用于数据读取测试
        if 'annos' in info:
            annos = info['annos']
            input_dict.update(annos)

        # 数据增强 (训练时开启)
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict