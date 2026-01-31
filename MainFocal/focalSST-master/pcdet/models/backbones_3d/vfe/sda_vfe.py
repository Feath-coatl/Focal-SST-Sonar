import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfe_template import VFETemplate

class PFNLayerWithAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        带有双重注意力机制的 Point Feature Network 层
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            use_norm: 是否使用 BatchNorm
            last_layer: 是否是最后一层 (最后一层通常做 MaxPool)
        """
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        
        # 1. 线性变换层 (Linear / Conv1d)
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            
        self.relu = nn.ReLU()

        # 2. 注意力模块 (从 TANet 提炼出的简化版双重注意力)
        # 目的：在 MaxPool 之前，让网络“看清”哪些点是噪声，哪些是目标
        if not self.last_layer:
            # Channel-wise Attention: 关注哪些特征通道重要
            self.ca_fc = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2, bias=False),
                nn.ReLU(),
                nn.Linear(out_channels // 2, out_channels, bias=False),
                nn.Sigmoid()
            )
            
            # Point-wise Attention: 关注体素内的哪些点重要 (抗噪关键)
            self.pa_fc = nn.Sequential(
                nn.Linear(out_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch*Voxels, Num_Points, In_Channels)
        """
        # Linear Mapping
        x = self.linear(inputs)  # [M, N, C]
        
        if self.use_norm:
            # BatchNorm1d 需要 (N, C, L) 格式，这里把 Num_Points 当作 Length
            # x.permute(0, 2, 1): [M, C, N]
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        x = self.relu(x)

        # Apply Attention (仅在前几层应用，最后一层聚合)
        if not self.last_layer:
            # 1. Channel Attention
            # 对点维度做 MaxPool -> [M, C]
            x_max, _ = torch.max(x, dim=1)
            ca_mask = self.ca_fc(x_max) # [M, C]
            ca_mask = ca_mask.unsqueeze(1) # [M, 1, C]
            
            # 2. Point Attention
            pa_mask = self.pa_fc(x) # [M, N, 1]
            
            # 3. 融合: 原始特征 * (1 + CA + PA)
            # 这里使用残差连接风格，防止梯度消失
            x = x * (1 + ca_mask + pa_mask)

        # Max Pooling Aggregation
        # 你的声呐数据每个体素内的点数可能很多，Max Pool 能有效提取最强反射
        x_max, _ = torch.max(x, dim=1)
        return x_max

class SDAVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        """
        Stacked Double Attention VFE
        """
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        # 读取配置中的参数
        # num_filters: 例如 [64, 128] 表示两层 PFN
        num_filters = self.model_cfg.NUM_FILTERS
        self.use_norm = self.model_cfg.get('USE_NORM', True)
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSLOTE_XYZ', True)

        # 初始输入通道数
        # x, y, z, intensity (4) + (x_center, y_center, z_center) (3) = 7
        if self.use_absolute_xyz:
            num_input_features = num_point_features + 3 
        else:
            num_input_features = num_point_features

        self.pfn_layers = nn.ModuleList()
        for i, out_filters in enumerate(num_filters):
            is_last_layer = (i == len(num_filters) - 1)
            pfn = PFNLayerWithAttention(
                in_channels=num_input_features,
                out_channels=out_filters,
                use_norm=self.use_norm,
                last_layer=is_last_layer
            )
            self.pfn_layers.append(pfn)
            num_input_features = out_filters

        self.num_point_features = num_filters[-1]

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: (num_voxels)
                voxel_coords: (num_voxels, 4) [batch_idx, z, y, x]
                ...
        """
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']
        
        # 1. 点云特征增强 (Add Mean Center)
        # 计算每个体素的几何中心
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
                      voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        
        # f_cluster = points - mean
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 构建输入特征向量
        # [x, y, z, intensity, x-xc, y-yc, z-zc]
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster]
        else:
            features = [voxel_features[..., 3:], f_cluster]

        features = torch.cat(features, dim=-1)

        # 2. 处理 Padding 部分 (Masking)
        # 由于 voxels 是定长的 tensor，点数不足的地方是 0。
        # 我们需要 mask 掉这些 0，防止它们影响 MaxPool
        mask = self.get_paddings_indicator(voxel_num_points, voxel_features.shape[1], axis=0)
        mask = mask.unsqueeze(-1).type_as(voxel_features)
        
        # 3. 通过 PFN Layers
        x = features
        for pfn in self.pfn_layers:
            x = pfn(x)
        
        # 将结果存回 batch_dict
        batch_dict['voxel_features'] = x
        
        return batch_dict