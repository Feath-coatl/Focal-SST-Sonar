import torch
import torch.nn as nn
from functools import partial

from ...utils import spconv_utils
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class FocalSparseEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] # [z, y, x] -> [x, y, z] usually in spconv? No, spconv uses [z, y, x] usually.
        
        # 确认 spconv 版本并获取基础模块
        self.spconv_ver = spconv_utils.get_spconv_ver()
        if self.spconv_ver == 1:
            raise NotImplementedError("Spconv 1.x is not supported for Focal Encoder")
        
        import spconv.pytorch as spconv
        
        # === 1. 基础稀疏卷积层 (Stem) ===
        # 将 VFE 输出的特征维度映射到主干维度 (e.g. 64 -> 64)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
            spconv.SubMConv3d(32, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )

        # === 2. Focal 预测头 (Mask Prediction) ===
        # 预测每个体素是否需要“生长”
        # 输入: 64 ch -> 输出: 1 ch (Logits)
        self.mask_head = nn.Linear(64, 1, bias=True)
        
        # === 3. 扩张卷积 (Dilation / Generation) ===
        # 对于被 Mask 选中的位置，我们需要生成新的特征。
        # 这里简化处理：我们对所有点进行一次 Kernel=3 的稀疏卷积，
        # 这会自动在所有邻域生成点 (spconv 特性)，然后我们利用 Mask 过滤掉不需要的点。
        self.conv_focal = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, 3, padding=1, bias=False, indice_key='focal_grow'), # standard conv expands
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.num_point_features = 64 # 输出给 DSVT 的通道数
        
        # Loss 配置
        self.loss_cfg = self.model_cfg.get('LOSS_CONFIG', {})
        self.focal_alpha = self.loss_cfg.get('ALPHA', 0.25)
        self.focal_gamma = self.loss_cfg.get('GAMMA', 2.0)
        self.mask_loss_weight = self.loss_cfg.get('LOSS_WEIGHT', 1.0)

    def get_output_feature_dim(self):
        return self.num_point_features

    def generate_mask_target(self, batch_dict, coords):
        """
        根据 GT Box 生成 Mask 标签。
        Args:
            batch_dict: 包含 gt_boxes
            coords: (N, 4) [batch_idx, z, y, x] 当前稀疏体素的坐标
        Returns:
            mask_target: (N,) 0 or 1
        """
        gt_boxes = batch_dict['gt_boxes'] # (B, M, 7+C)
        batch_size = len(gt_boxes)
        
        # 将体素坐标转换为真实物理坐标 (用于和 GT Box 比较)
        # 注意：这里需要知道 Voxel Size 和 Point Cloud Range
        # 假设我们能从 batch_dict 获取或通过配置传入。
        # 为简化，这里假设 data_dict 已经有了 'voxel_features' 对应的 point_coords
        # 如果没有，我们需要重新计算。通常 spconv 不保存物理坐标。
        
        # 简化策略：
        # 我们暂时假设输入特征是 VFE 出来的，VFE 应该保留了 'voxel_coords'。
        # 我们需要从 configs 读取 voxel_size
        voxel_size = torch.tensor(self.model_cfg.VOXEL_SIZE, device=coords.device)
        pc_range = torch.tensor(self.model_cfg.POINT_CLOUD_RANGE, device=coords.device)
        
        # indices (B, Z, Y, X) -> metric (X, Y, Z)
        # coords[:, 0] is batch_idx
        spatial_indices = coords[:, 1:].float() # Z, Y, X
        
        # 转换回 X, Y, Z 格式 (OpenPCDet 标准: ZYX indices -> XYZ coords)
        # x = x_idx * vx + min_x + vx/2
        spatial_coords = spatial_indices[:, [2, 1, 0]] * voxel_size + pc_range[:3] + voxel_size / 2
        
        mask_target_list = []
        
        for k in range(batch_size):
            mask = coords[:, 0] == k
            cur_coords = spatial_coords[mask] # (N_k, 3)
            cur_gt = gt_boxes[k] # (M, 7)
            
            # 过滤掉全0的 padding box
            cur_gt = cur_gt[cur_gt[:, 3] > 0]
            
            if len(cur_gt) == 0:
                mask_target_list.append(torch.zeros(cur_coords.shape[0], dtype=torch.float, device=coords.device))
                continue
                
            # 检查点是否在 Box 内
            # 使用 OpenPCDet 内置 CUDA op 加速
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                cur_coords.unsqueeze(0), cur_gt[:, :7].unsqueeze(0)
            ).long().squeeze(0) # (N_k)
            
            # box_idxs_of_pts == -1 表示不在任何框内 (背景)
            # 我们将 Box 内的点设为 1 (需要保留/生长)，Box 外设为 0
            label = (box_idxs_of_pts >= 0).float()
            mask_target_list.append(label)
            
        return torch.cat(mask_target_list, dim=0)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxel_features, voxel_coords
        """
        import spconv.pytorch as spconv
        
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # 1. 构建 Sparse Tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # 2. 基础特征提取 (SubM Conv)
        # 保持稀疏性不变，只提特征
        x = self.conv_input(input_sp_tensor)
        
        # 3. Mask 预测 (Focal 部分)
        # 我们基于当前有的体素，预测它是“重要目标”还是“背景噪声”
        # 这里的 Mask 用于计算 Loss，辅助主干网络聚焦
        if self.training:
            mask_logits = self.mask_head(x.features) # (N, 1)
            mask_prob = torch.sigmoid(mask_logits.view(-1))
            
            # 生成监督信号
            mask_target = self.generate_mask_target(batch_dict, x.indices)
            
            # 计算 Focal Loss
            # p_t = p if y=1 else 1-p
            p_t = mask_prob * mask_target + (1 - mask_prob) * (1 - mask_target)
            loss = - self.focal_alpha * (1 - p_t) ** self.focal_gamma * torch.log(p_t + 1e-6)
            
            # 存入 batch_dict，后续在 Model 层面汇总
            # 注意：OpenPCDet 的 loss 通常是在 Head 里算的，
            # 如果这里是 Backbone，我们需要把 loss 挂载到 tb_dict (tensorboard) 中
            focal_loss = loss.mean() * self.mask_loss_weight
            
            if 'loss_box_of_backbone' not in batch_dict:
                batch_dict['loss_box_of_backbone'] = 0
            batch_dict['loss_box_of_backbone'] += focal_loss

        # 4. 结构增强 (Dilation)
        # 简单的做法：使用 kernel=3 的标准 SparseConv，它会使非空点周围的空点变非空 (Dilation 效果)
        # 复杂的 Focal Conv 会根据 Mask 筛选这些新生成的点。
        # 考虑到声呐极其稀疏，我们先全量 Dilation，让 DSVT 去处理。
        # 如果显存不够，后续再加 Top-K 筛选逻辑。
        x_dilated = self.conv_focal(x)
        
        # 将结果传给下一层 (DSVT)
        # DSVT 需要的是 Sparse Tensor
        batch_dict['encoded_spconv_tensor'] = x_dilated
        batch_dict['encoded_spconv_tensor_stride'] = 1 # 我们没做下采样

        return batch_dict