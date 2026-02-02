import torch
import torch.nn as nn
from functools import partial

from ...utils import spconv_utils
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import split_voxels, check_repeat, FocalLoss

class FocalSparseEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] # [z, y, x] -> [x, y, z] usually in spconv? No, spconv usually uses [z, y, x] as spatial shape.
        
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

        # === 2. Focal 预测头 (Importance Prediction) ===
        # 原实现使用 Linear，现改为 SubMConv3d 以利用局部上下文信息
        # 输入: 64 ch -> 输出: 27 ch (26个kernel位置 + 1个voxel中心的importance)
        # kernel_offsets有26个位置(3x3x3-1), 加上中心voxel共27个
        self.conv_imp = spconv.SubMConv3d(64, 27, kernel_size=3, stride=1, padding=1, bias=False, indice_key='imp')

        # === 3. Focal 处理层 (Focal Processing) ===
        # 这里的卷积层用于处理 Split+Dilation 之后的特征
        # 由于 split_voxels 已经完成了物理上的"膨胀"（生成了邻域坐标），这里只需用 SubMConv 进行特征提取即可
        self.conv_focal = spconv.SparseSequential(
            spconv.SubMConv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, indice_key='focal'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 128 # 输出给 DSVT 的通道数
        
        # === 配置参数 ===
        self.loss_cfg = self.model_cfg.get('LOSS_CONFIG', {})
        # 使用导入的 FocalLoss 或保留自定义逻辑，这里保留你原有的 Loss 计算逻辑以适配 Box 监督
        self.focal_loss_func = FocalLoss() 
        self.focal_alpha = self.loss_cfg.get('ALPHA', 0.25)
        self.focal_gamma = self.loss_cfg.get('GAMMA', 2.0)
        self.mask_loss_weight = self.loss_cfg.get('LOSS_WEIGHT', 1.0)
        
        self.threshold = self.model_cfg.get('FOCAL_THRESHOLD', 0.5)
        self.topk = self.model_cfg.get('FOCAL_TOPK', False) # 是否使用 topk 筛选
        self.mask_multi = False # 是否对特征乘 mask kernel，通常设为 False 即可

        # === split_voxels 需要的辅助张量 ===
        # 3x3x3 膨胀核的偏移量
        _step = 1 # kernel_size // 2
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.register_buffer('kernel_offsets', torch.Tensor(kernel_offsets))
        # 坐标映射: spconv indices (b, z, y, x) -> metric coords (x, y, z) 需要用到 inv_idx
        self.register_buffer('inv_idx', torch.Tensor([2, 1, 0]).long())
        
        # 从配置中读取体素大小和范围，用于坐标转换
        self.voxel_size = torch.tensor(self.model_cfg.VOXEL_SIZE).cuda()
        self.point_cloud_range = torch.tensor(self.model_cfg.POINT_CLOUD_RANGE).cuda()

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
        # indices (B, Z, Y, X) -> metric (X, Y, Z)
        spatial_indices = coords[:, 1:].float() # Z, Y, X
        
        # 转换回 X, Y, Z 格式 (OpenPCDet 标准: ZYX indices -> XYZ coords)
        spatial_coords = spatial_indices[:, [2, 1, 0]] * self.voxel_size + self.point_cloud_range[:3] + self.voxel_size / 2
        
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
        x = self.conv_input(input_sp_tensor)
        
        # 3. Mask 预测 (Focal 部分)
        # 计算 Importance Score
        x_imp = self.conv_imp(x)
        imps_3d = x_imp.features # (N, 27) Logits: 26个kernel位置 + 1个voxel中心
        mask_prob = torch.sigmoid(imps_3d[:, -1])  # 只使用最后一个通道(voxel importance)进行监督
        
        if self.training:
            # 生成监督信号
            mask_target = self.generate_mask_target(batch_dict, x.indices)
            
            # 计算 Focal Loss
            p_t = mask_prob * mask_target + (1 - mask_prob) * (1 - mask_target)
            loss = - self.focal_alpha * (1 - p_t) ** self.focal_gamma * torch.log(p_t + 1e-6)
            focal_loss = loss.mean() * self.mask_loss_weight
            
            if 'loss_box_of_backbone' not in batch_dict:
                batch_dict['loss_box_of_backbone'] = 0
            batch_dict['loss_box_of_backbone'] += focal_loss

        # 4. Focal Split & Dilation (核心逻辑修改)
        # 准备 split_voxels 需要的参数
        # 计算 Metric Coords: [N, 3] (x, y, z)
        spatial_indices = x.indices[:, 1:].float()
        voxels_3d = spatial_indices[:, [2, 1, 0]] * self.voxel_size + self.point_cloud_range[:3]
        
        batch_dict_mock = {'gt_boxes': batch_dict.get('gt_boxes', None)} # split_voxels 内部可能用到 gt_boxes 进行 debug，虽然主要逻辑不需要

        # 使用 CUDA 算子进行 Split
        # features_fore: 前景特征 (包含膨胀后的点)
        # indices_fore: 前景坐标 (包含膨胀后的坐标)
        # features_back: 背景特征 (仅保留原有的点)
        # indices_back: 背景坐标
        features_fore_list = []
        indices_fore_list = []
        features_back_list = []
        indices_back_list = []
        
        # 由于 split_voxels 是按 batch 循环的逻辑设计的（参考 focal_sparse_conv.py），这里模拟循环
        # 注意：split_voxels 内部实现可能依赖 batch_idx 逐个处理
        
        for b in range(batch_size):
            # 提取当前 batch 的数据
            batch_mask = x.indices[:, 0] == b
            # 注意：split_voxels 需要传入的是全量的 tensor，它内部会根据 b 进行筛选
            # 但 imps_3d 和 voxels_3d 必须和 x 对齐
            
            # 调用 split_voxels
            # mask_multi=False (我们通常不需要对特征进行加权，仅做筛选和生成)
            f_fore, i_fore, f_back, i_back, _ = split_voxels(
                x, b, imps_3d, voxels_3d, self.kernel_offsets, 
                mask_multi=self.mask_multi, topk=self.topk, threshold=self.threshold
            )
            
            features_fore_list.append(f_fore)
            indices_fore_list.append(i_fore)
            features_back_list.append(f_back)
            indices_back_list.append(i_back)

        # 合并列表
        features_fore = torch.cat(features_fore_list, dim=0)
        indices_fore = torch.cat(indices_fore_list, dim=0)
        features_back = torch.cat(features_back_list, dim=0)
        indices_back = torch.cat(indices_back_list, dim=0)
        
        # 5. Combine & Check Repeat (合并前景背景)
        # 将前景和背景拼接
        x_merged_features = torch.cat([features_fore, features_back], dim=0)
        x_merged_indices = torch.cat([indices_fore, indices_back], dim=0)
        
        # 使用 check_repeat 去除重复点 (因为前景膨胀可能覆盖到原本的背景点，或者不同前景点的膨胀区域重叠)
        # check_repeat 会对坐标进行哈希去重
        features_out_list = []
        indices_out_list = []
        
        # check_repeat 也需要逐 batch 处理以保证正确性
        for b in range(batch_size):
            batch_mask = x_merged_indices[:, 0] == b
            if batch_mask.sum() == 0:
                continue
            f_out, i_out, _ = check_repeat(
                x_merged_features[batch_mask], x_merged_indices[batch_mask], flip_first=False
            )
            features_out_list.append(f_out)
            indices_out_list.append(i_out)
            
        if len(features_out_list) > 0:
            x_final_features = torch.cat(features_out_list, dim=0)
            x_final_indices = torch.cat(indices_out_list, dim=0)
        else:
            # 极少数情况（全空），保持原样或返回空
            x_final_features = x.features
            x_final_indices = x.indices

        # 构建新的 SparseTensor
        x_focal = spconv.SparseConvTensor(
            features=x_final_features,
            indices=x_final_indices,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # 6. Focal Feature Extraction
        # 对致密化后的结构进行卷积提取
        x_out = self.conv_focal(x_focal)
        
        # 将结果传给下一层 (DSVT)
        batch_dict['encoded_spconv_tensor'] = x_out
        batch_dict['encoded_spconv_tensor_stride'] = 1 

        return batch_dict