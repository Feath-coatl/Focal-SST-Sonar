"""
FocalToVoxelNeXtBridge: 将 FocalDSVT 的输出转换为 VoxelNeXt 格式的稀疏 2D Tensor。

FocalDSVT 输出:
    - batch_dict['voxel_features']: (N, C) 体素特征
    - batch_dict['voxel_coords']: (N, 4) [batch_idx, z, y, x]

VoxelNeXtHead 期望:
    - batch_dict['encoded_spconv_tensor']: SparseConvTensor (稀疏2D)
        - indices: (M, 3) -> [batch_idx, y, x]
        - features: (M, C)
        - spatial_shape: [Y, X]
"""

import torch
import torch.nn as nn
from functools import partial

try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv


class FocalToVoxelNeXtBridge(nn.Module):
    """
    将 FocalDSVT 的 pillar 特征转换为 VoxelNeXt 格式的稀疏 2D Tensor。
    使用累加聚合（sum pooling）而非 max pooling 以保留更多高度信息。
    """
    def __init__(self, model_cfg, grid_size, voxel_size=None, point_cloud_range=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = model_cfg.get('INPUT_CHANNELS', 128)
        self.num_bev_features = model_cfg.get('NUM_BEV_FEATURES', 128)
        
        # 存储配置用于计算spatial_shape
        self.grid_size = grid_size  # [X, Y, Z]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # 特征映射（如果INPUT_CHANNELS != NUM_BEV_FEATURES）
        if self.input_channels != self.num_bev_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(self.input_channels, self.num_bev_features),
                nn.BatchNorm1d(self.num_bev_features),
                nn.ReLU()
            )
        else:
            self.feature_proj = nn.Identity()
        
        # 可选：添加额外的稀疏卷积进行特征增强
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(self.num_bev_features, self.num_bev_features, 3, 
                            stride=1, padding=1, bias=True, indice_key='bridge_subm'),
            norm_fn(self.num_bev_features),
            nn.ReLU(True),
        )
        
        # 计算BEV的spatial_shape（基于grid_size的X和Y维度）
        # grid_size 是 [X_num, Y_num, Z_num]
        self.bev_spatial_shape = [grid_size[1], grid_size[0]]  # [Y, X] for spconv
    
    def bev_aggregate(self, features, indices, batch_size):
        """
        将3D稀疏特征聚合到BEV平面，类似VoxelNeXt的bev_out。
        使用feature累加（sum pooling）而非max pooling以保留更多信息。
        
        Args:
            features: (N, C) 体素特征
            indices: (N, 4) [batch, z, y, x]
            batch_size: int
            
        Returns:
            unique_features: (M, C) 聚合后的BEV特征
            unique_bev_indices: (M, 3) [batch, y, x]
        """
        # indices: [N, 4] -> [batch, z, y, x]
        # 提取BEV索引: [batch, y, x]
        bev_indices = indices[:, [0, 2, 3]]
        
        # 找到unique的BEV位置
        unique_bev_indices, inverse_indices = torch.unique(
            bev_indices, dim=0, return_inverse=True
        )
        
        # 累加聚合特征
        unique_features = features.new_zeros((unique_bev_indices.shape[0], features.shape[1]))
        unique_features.index_add_(0, inverse_indices, features)
        
        # 可选：计数并平均（mean pooling），取消下面的注释即可使用
        # counts = features.new_zeros(unique_bev_indices.shape[0]).index_add_(
        #     0, inverse_indices, torch.ones(features.shape[0], device=features.device)
        # )
        # unique_features = unique_features / counts.unsqueeze(-1).clamp(min=1)
        
        return unique_features, unique_bev_indices

    def forward(self, batch_dict):
        """
        转换FocalDSVT输出为VoxelNeXt兼容格式。
        
        Args:
            batch_dict: 包含 voxel_features, voxel_coords, batch_size
            
        Returns:
            batch_dict: 添加了 encoded_spconv_tensor
        """
        # 获取DSVT输出的pillar特征
        voxel_features = batch_dict['voxel_features']  # (N, C)
        voxel_coords = batch_dict['voxel_coords']       # (N, 4) [batch, z, y, x]
        batch_size = batch_dict['batch_size']
        
        # 特征投影
        features = self.feature_proj(voxel_features)
        
        # BEV聚合
        bev_features, bev_indices = self.bev_aggregate(features, voxel_coords, batch_size)
        
        # 转换为int32 (spconv要求)
        bev_indices = bev_indices.int()
        
        # 创建稀疏2D Tensor
        sp_tensor = spconv.SparseConvTensor(
            features=bev_features,
            indices=bev_indices,
            spatial_shape=self.bev_spatial_shape,
            batch_size=batch_size
        )
        
        # 应用shared_conv（与VoxelNeXt一致）进行特征增强
        sp_tensor = self.shared_conv(sp_tensor)
        
        # 更新batch_dict
        batch_dict['encoded_spconv_tensor'] = sp_tensor
        batch_dict['encoded_spconv_tensor_stride'] = 1  # FocalDSVT通常stride=1
        
        return batch_dict
