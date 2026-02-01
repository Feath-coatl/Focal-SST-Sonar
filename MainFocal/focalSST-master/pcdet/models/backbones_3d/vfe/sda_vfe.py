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
        
        # 1. 线性变换部分保持不变
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            
        self.relu = nn.ReLU()

        # 2. 注意力模块 (DA: Double Attention)
        # 注意：中间层也需要 Attention，但不需要 MaxPool
        self.ca_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels // 2, out_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.pa_fc = nn.Sequential(
            nn.Linear(out_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch*Voxels, Num_Points, In_Channels) -> [M, N, C_in]
        """
        # Linear Mapping
        x = self.linear(inputs)  # [M, N, C_out]
        
        if self.use_norm:
            # BatchNorm1d 需要 (M, C, N)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        x = self.relu(x)

        # --- Double Attention ---
        # 1. Point-wise Attention [M, N, 1]
        pa_weight = self.pa_fc(x)
        
        # 2. Channel-wise Attention [M, 1, C]
        # 先临时 MaxPool 得到全局特征用于计算通道权重
        x_global, _ = torch.max(x, dim=1, keepdim=True) # [M, 1, C]
        ca_weight = self.ca_fc(x_global)
        
        # 3. 融合 (参考 TANet: 乘法融合)
        # 两个权重相乘，得到最终的 Attention Mask
        combined_attention = pa_weight * ca_weight 
        
        # 应用 Attention (TANet 风格)
        # x_attended = x * combined_attention
        # 或者保留你的残差风格 (更易训练):
        x = x * (1 + combined_attention)

        # --- 关键修复：维度保持 ---
        if self.last_layer:
            # 只有最后一层才把点聚合 (M, N, C) -> (M, C)
            x_max, _ = torch.max(x, dim=1)
            return x_max
        else:
            # 中间层返回包含点维度的特征，供下一层继续提取 Point Attention
            return x

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

        # [新增] 读取体素尺寸和点云范围，用于计算几何中心
        # 格式: [vx, vy, vz] 和 [xmin, ymin, zmin, xmax, ymax, zmax]
        self.voxel_size = torch.tensor(self.model_cfg.VOXEL_SIZE).cuda()
        self.point_cloud_range = torch.tensor(self.model_cfg.POINT_CLOUD_RANGE).cuda()

        # 初始输入通道数
        # x, y, z, intensity (4) 
        # + (x_mean, y_mean, z_mean) (3)
        # + (x_center, y_center, z_center) (3) = 10
        if self.use_absolute_xyz:
            num_input_features = num_point_features + 3 + 3
        else:
            num_input_features = num_point_features + 3

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
        
        # 2. [新增] 计算几何中心偏移 (f_center)
        # coords: [batch_idx, z, y, x]
        # 注意: OpenPCDet 坐标顺序通常是 z, y, x
        z_idx = coords[:, 1].float()
        y_idx = coords[:, 2].float()
        x_idx = coords[:, 3].float()
        
        # 计算几何中心 (xyz)
        # center = idx * size + range_min + size/2
        # self.voxel_size: [vx, vy, vz]
        # self.point_cloud_range: [xmin, ymin, zmin, ...]
        
        x_center = x_idx * self.voxel_size[0] + self.point_cloud_range[0] + self.voxel_size[0] / 2
        y_center = y_idx * self.voxel_size[1] + self.point_cloud_range[1] + self.voxel_size[1] / 2
        z_center = z_idx * self.voxel_size[2] + self.point_cloud_range[2] + self.voxel_size[2] / 2
        
        # Stack 并在点维度广播
        # center: (M, 3) -> (M, 1, 3)
        center = torch.stack([x_center, y_center, z_center], dim=-1).unsqueeze(1)
        f_center = voxel_features[:, :, :3] - center.type_as(voxel_features)

        # 3. 构建输入特征 (10 dim)
        # [x, y, z, i, x-xm, y-ym, z-zm, x-xc, y-yc, z-zc]
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        features = torch.cat(features, dim=-1)

        # 4. 处理 Padding 部分 (Masking)
        voxel_count = features.shape[1]
        # 由于 voxels 是定长的 tensor，点数不足的地方是 0。
        # 我们需要 mask 掉这些 0，防止它们影响 MaxPool
        mask = self.get_paddings_indicator(voxel_num_points, voxel_features.shape[1], axis=0)
        mask = mask.unsqueeze(-1).type_as(voxel_features)
        features *= mask

        # 5. 通过 PFN Layers
        x = features
        for pfn in self.pfn_layers:
            x = pfn(x)
        
        # 将结果存回 batch_dict
        batch_dict['voxel_features'] = x
        
        return batch_dict