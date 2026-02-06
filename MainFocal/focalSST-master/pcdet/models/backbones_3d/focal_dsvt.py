import torch
import torch.nn as nn
from .focal_sparse_encoder import FocalSparseEncoder
from .dsvt import DSVT

class FocalDSVT(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        """
        Focal SST 的核心组合主干：Focal Sparse Encoder + DSVT
        """
        super().__init__()
        self.model_cfg = model_cfg
        
        # 1. 初始化第一阶段：Focal Sparse Encoder (负责去稀疏、补全结构)
        # 它的配置在 yaml 的 FOCAL_ENCODER 字段下
        self.focal_encoder = FocalSparseEncoder(
            model_cfg=model_cfg.FOCAL_ENCODER,
            input_channels=input_channels,
            grid_size=grid_size,
            **kwargs
        )
        
        # Focal Encoder 的输出通道数，将作为 DSVT 的输入
        focal_out_channels = self.focal_encoder.get_output_feature_dim()

        # 2. 初始化第二阶段：DSVT (负责长程上下文建模)
        # 它的配置在 yaml 的 DSVT 字段下
        self.dsvt = DSVT(
            model_cfg=model_cfg.DSVT,
            input_channels=focal_out_channels,
            grid_size=grid_size,
            **kwargs
        )
        
        self.num_point_features = self.dsvt.num_point_features
        
        # 保存DSVT的sparse_shape配置，用于后续创建SparseTensor
        # DSVT的sparse_shape应该是[Z, Y, X]格式 (spconv standard)
        self.dsvt_sparse_shape = model_cfg.DSVT.INPUT_LAYER.get('sparse_shape', grid_size[::-1])

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict):
        # 1. 执行 Focal Encoder
        # 这会更新 batch_dict['encoded_spconv_tensor']
        batch_dict = self.focal_encoder(batch_dict)
        
        # === 关键桥接逻辑 ===
        # DSVT 通常设计为从 VFE 接收原始 voxel_features 和 voxel_coords。
        # 这里我们需要“欺骗”DSVT，把 Focal Encoder 生成的、更致密的稀疏特征
        # 当作是“新的体素输入”传给它。
        
        sp_tensor = batch_dict['encoded_spconv_tensor']
        
        # 将 SparseTensor 拆解回 features 和 coords
        # 这样 DSVT 的 input_layer 就能像处理普通体素一样处理它
        batch_dict['voxel_features'] = sp_tensor.features
        batch_dict['voxel_coords'] = sp_tensor.indices
        
        # 注意：DSVT 需要知道当前的 voxel 数量
        # batch_dict['voxel_num_points'] 在这里已经失效了，因为这是经过卷积后的特征
        # 但 DSVT 内部计算主要是基于坐标的，通常不强依赖 voxel_num_points (除非用 pillar vfe)
        # 我们安全起见，可以不更新 voxel_num_points，或者设为 None
        
        # 2. 执行 DSVT
        batch_dict = self.dsvt(batch_dict)
        
        return batch_dict

