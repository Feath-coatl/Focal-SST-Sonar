import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfe_template import VFETemplate

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        简化的 Point Feature Network 层 (纯 MLP)
        结构: Linear -> BN -> ReLU
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            use_norm: 是否使用 BatchNorm
            last_layer: 是否是最后一层 (用于标记，SimpleVFE通常在所有层后统一做MaxPool，但为了兼容性保留此参数)
        """
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        
        # 纯 MLP 结构
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch*Voxels, Num_Points, In_Channels)
        """
        x = self.linear(inputs)
        
        if self.use_norm:
            # BatchNorm1d 需要 (N, C, L) 格式
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        x = self.relu(x)
        return x

class SimpleVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        """
        Simple Voxel Feature Encoder (MLP-based)
        支持 7维/10维 特征扩展，用于与 SDAVFE 进行公平对比
        """
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        # === 1. 参数解析 ===
        # 读取配置，保持与 SDAVFE 接口一致
        num_filters = self.model_cfg.NUM_FILTERS # e.g. [64, 64]
        self.use_norm = self.model_cfg.get('USE_NORM', True)
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSLOTE_XYZ', True)
        self.with_distance = self.model_cfg.get('WITH_DISTANCE', False)
        
        # 读取几何参数 (用于 10维扩展)
        self.voxel_size = torch.tensor(self.model_cfg.get('VOXEL_SIZE', [0.1, 0.1, 0.1])).cuda()
        self.point_cloud_range = torch.tensor(self.model_cfg.get('POINT_CLOUD_RANGE', [0, -40, -3, 70.4, 40, 1])).cuda()

        # === 2. 输入通道计算 ===
        # 基础: x, y, z, intensity (4)
        num_input_features = num_point_features
        
        # 扩展 1: 均值偏移 (x-xm, y-ym, z-zm) -> +3
        num_input_features += 3 
        
        # 扩展 2: 绝对坐标/中心偏移 (x-xc, y-yc, z-zc) -> +3
        if self.use_absolute_xyz:
            num_input_features += 3
            
        # 扩展 3: 距离 (sqrt(x^2+y^2+z^2)) -> +1 (如果启用)
        if self.with_distance:
            num_input_features += 1

        # === 3. 构建 PFN 层堆叠 ===
        self.pfn_layers = nn.ModuleList()
        for i, out_filters in enumerate(num_filters):
            is_last_layer = (i == len(num_filters) - 1)
            pfn = PFNLayer(
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
        """
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']
        
        # === 1. 特征扩充 (与 SDAVFE 保持一致) ===
        # 1.1 计算均值中心偏移 f_cluster (x-xm, y-ym, z-zm)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
                      torch.clamp_min(voxel_num_points.view(-1, 1, 1), min=1.0).type_as(voxel_features)
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 1.2 构建基础特征列表
        feature_list = [voxel_features] # [x, y, z, i]
        feature_list.append(f_cluster)  # [x-xm, y-ym, z-zm]

        # 1.3 计算几何中心偏移 f_center (x-xc, y-yc, z-zc)
        if self.use_absolute_xyz:
            z_idx = coords[:, 1].float()
            y_idx = coords[:, 2].float()
            x_idx = coords[:, 3].float()
            
            # 计算几何中心 (xyz)
            x_center = x_idx * self.voxel_size[0] + self.point_cloud_range[0] + self.voxel_size[0] / 2
            y_center = y_idx * self.voxel_size[1] + self.point_cloud_range[1] + self.voxel_size[1] / 2
            z_center = z_idx * self.voxel_size[2] + self.point_cloud_range[2] + self.voxel_size[2] / 2
            
            center = torch.stack([x_center, y_center, z_center], dim=-1).unsqueeze(1)
            f_center = voxel_features[:, :, :3] - center.type_as(voxel_features)
            feature_list.append(f_center)

        # 1.4 计算距离 (如果配置启用)
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            feature_list.append(points_dist)

        # 拼接所有特征 (例如 4+3+3 = 10维)
        features = torch.cat(feature_list, dim=-1)

        # === 2. Masking (处理 Padding) ===
        # 将填充的 0 点特征置零，防止影响计算（虽然 MLP 对 0 输入输出通常不为 0 (bias/bn)，但后续 MaxPool 会处理）
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = mask.unsqueeze(-1).type_as(voxel_features)
        features *= mask
        
        # === 3. MLP 处理 ===
        x = features
        for pfn in self.pfn_layers:
            x = pfn(x)
        
        # === 4. Max Pooling Aggregation ===
        # 在点维度 (dim=1) 取最大值，得到体素特征
        # x: (M, N, C_out) -> (M, C_out)
        x_max, _ = torch.max(x, dim=1)
        
        batch_dict['voxel_features'] = x_max
        
        return batch_dict