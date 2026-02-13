"""
DsvtToDenseBEV: 将DSVT的输出(voxel_features + voxel_coords)
直接转换为Dense 2D BEV特征图。

用途: 消融实验中，使用与VoxelNeXt管线相同的DSVT特征，
      但通过dense BEV pipeline (BaseBEVBackbone + CenterHead) 进行检测。
      确保消融实验的变量控制仅在"检测管线"层面。

与现有MAP_TO_BEV模块的区别:
  - SparseToDenseDirect: 读取 encoded_spconv_tensor (FocalEncoder输出,pre-DSVT)
  - FocalToVoxelNeXtBridge: 读取 voxel_features → 输出稀疏2D tensor
  - DsvtToDenseBEV (本模块): 读取 voxel_features → 输出稠密2D tensor (spatial_features)
"""

import torch
import torch.nn as nn


class DsvtToDenseBEV(nn.Module):
    """
    将DSVT输出的稀疏体素特征通过max-pooling沿Z轴压缩为Dense BEV特征图。
    
    输入: batch_dict['voxel_features'] (N, C), batch_dict['voxel_coords'] (N, 4) [batch, z, y, x]
    输出: batch_dict['spatial_features'] (B, C, Y, X) — dense BEV特征
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
        # grid_size = [X_num, Y_num, Z_num]
        self.nx = grid_size[0]
        self.ny = grid_size[1]

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']   # (N, C)
        voxel_coords = batch_dict['voxel_coords']       # (N, 4) [batch, z, y, x]
        batch_size = batch_dict['batch_size']
        C = voxel_features.shape[1]

        # 初始化dense BEV tensor
        spatial_features = voxel_features.new_zeros((batch_size, C, self.ny, self.nx))

        # 逐batch填充: 对同一(y,x)位置的多个z值取max
        batch_idx = voxel_coords[:, 0].long()
        y_idx = voxel_coords[:, 2].long()
        x_idx = voxel_coords[:, 3].long()

        for b in range(batch_size):
            mask = (batch_idx == b)
            if mask.sum() == 0:
                continue
            
            b_features = voxel_features[mask]  # (M, C)
            b_y = y_idx[mask]
            b_x = x_idx[mask]
            
            # 使用线性索引 + scatter_reduce 实现高效的max-pooling
            linear_idx = b_y * self.nx + b_x  # (M,)
            
            # 创建独立的临时tensor，避免view导致的inplace冲突
            flat_spatial = torch.zeros(C, self.ny * self.nx, 
                                       dtype=voxel_features.dtype, 
                                       device=voxel_features.device)
            
            # scatter_reduce (非inplace): 对相同(y,x)位置取max
            linear_idx_expand = linear_idx.unsqueeze(0).expand(C, -1)  # (C, M)
            flat_spatial = torch.scatter_reduce(
                flat_spatial, 1, linear_idx_expand, b_features.t(), reduce='amax', include_self=False
            )
            spatial_features[b] = flat_spatial.view(C, self.ny, self.nx)

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = 1
        return batch_dict
