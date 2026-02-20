"""
VoxelNeXtHeadSonar: 适配Sonar数据的VoxelNeXt稀疏检测头。

主要改动（相比原版VoxelNeXtHead）：
1. 移除velocity预测分支（Sonar数据没有速度信息）
2. 添加数值稳定性处理（防止NaN）
3. 适配Sonar的类别和参数

参考: VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking (CVPR 2023)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
import copy

from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils

try:
    from ...ops.iou3d_nms import iou3d_nms_utils
except ImportError:
    iou3d_nms_utils = None

try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv


class SparseSeparateHead(nn.Module):
    """
    稀疏分支预测头，使用SubMConv2d替代普通Conv2d。
    每个预测分支（hm, center, center_z, dim, rot）独立预测。
    """
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, 
                                     padding=int(kernel_size//2), bias=use_bias, 
                                     indice_key=f'{cur_name}_{k}'),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, 
                                            bias=True, indice_key=f'{cur_name}_out'))
            fc = nn.Sequential(*fc_list)
            
            # 正确初始化所有层
            for m in fc.modules():
                if isinstance(m, spconv.SubMConv2d):
                    kaiming_normal_(m.weight.data)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            # 热图分支：最后一层bias初始化为负值，使初始sigmoid输出接近0（偏向背景）
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        """
        Args:
            x: SparseConvTensor
        Returns:
            ret_dict: 包含各分支预测的字典
        """
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features
        return ret_dict


class VoxelNeXtHeadSonar(nn.Module):
    """
    适配Sonar数据的VoxelNeXt稀疏检测头。
    
    核心特点：
    1. 使用稀疏卷积SubMConv2d替代Dense Conv2d
    2. 目标分配基于稀疏体素（而非Dense feature map）
    3. 直接回归Z坐标（center_z），支持垂直方向检测
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, 
                 point_cloud_range, voxel_size, predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', 1)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])

        # 解析每个head负责的类别
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        # 构建各个Head
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SparseSeparateHead(
                    input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128),
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )
        
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        """构建稀疏损失函数"""
        self.add_module('hm_loss_func', FocalLossSparse())
        self.add_module('reg_loss_func', RegLossSparse())
        
        # IoU回归损失（v4新增）：通过配置中HEAD_DICT包含'iou'键来启用
        self.use_iou = 'iou' in self.separate_head_cfg.HEAD_DICT
        if self.use_iou:
            self.iou_loss_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('iou_weight', 0.5)
            # IoU rectify系数：final_score = hm_score^(1-w) * iou_pred^w
            self.iou_rectify_weight = self.model_cfg.LOSS_CONFIG.get('IOU_RECTIFY_WEIGHT', 0.5)

    def distance(self, voxel_indices, center):
        """
        计算体素到中心的欧氏距离平方。
        
        Args:
            voxel_indices: (N, 2) 体素索引 [y, x]
            center: (2,) 目标中心 [y, x] (注意：统一使用y,x格式)
            
        Returns:
            distances: (N,) 距离平方
        """
        # 确保坐标格式一致 [y, x]
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape, 
            feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2
    ):
        """
        为单个检测头分配目标。
        
        Args:
            num_classes: 类别数
            gt_boxes: (N, 8) 真实框 [x, y, z, dx, dy, dz, heading, class_id]
            num_voxels: int, 当前batch的体素数量
            spatial_indices: (num_voxels, 2) 体素的[y, x]索引
            spatial_shape: [Y, X] BEV尺寸
            feature_map_stride: 特征图步长
            num_max_objs: 最大目标数
            gaussian_overlap: 高斯核重叠率
            min_radius: 最小半径
        
        Returns:
            heatmap: (num_classes, num_voxels) 稀疏热图
            ret_boxes: (num_max_objs, code_size) 回归目标
            inds: (num_max_objs,) 目标体素索引
            mask: (num_max_objs,) 有效mask
        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        # 边界情况：无GT或无体素
        if gt_boxes.shape[0] == 0 or num_voxels == 0:
            return heatmap, ret_boxes, inds, mask

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        
        # 计算GT在特征图上的坐标
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)
        # 重要修复：统一使用 [y, x] 格式以匹配 spatial_indices
        center = torch.cat((coord_y[:, None], coord_x[:, None]), dim=-1)  # [y, x] 格式

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            # center[k] 现在是 [y, x] 格式
            if not (0 <= center[k][1] <= spatial_shape[1] and 0 <= center[k][0] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            # 确保class_id有效
            if cur_class_id < 0 or cur_class_id >= num_classes:
                continue
            
            # 计算每个体素到GT中心的距离 (坐标格式现在一致了)
            distance = self.distance(spatial_indices.float(), center[k])
            inds[k] = distance.argmin()  # 最近的体素
            mask[k] = 1

            # 绘制高斯热图
            if 'gt_center' in self.gaussian_type:
                self._draw_gaussian_to_heatmap_voxels(
                    heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio
                )

            if 'nearst' in self.gaussian_type:
                self._draw_gaussian_to_heatmap_voxels(
                    heatmap[cur_class_id], 
                    self.distance(spatial_indices.float(), spatial_indices[inds[k]].float()),
                    radius[k].item() * self.gaussian_ratio
                )

            # 关键修复：确保最近体素的热图值为1（保证有正样本参与训练）
            # 由于高斯衰减，即使最近体素的值也可能<1，导致pos_inds=0
            heatmap[cur_class_id, inds[k]] = 1.0

            # 编码回归目标 (注意：offset也使用[y,x]格式，即[offset_y, offset_x])
            # ret_boxes[:, 0:2] = [offset_y, offset_x]
            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2].float()
            ret_boxes[k, 2] = z[k]  # 直接回归z坐标
            # 添加数值稳定性：防止log(0)
            ret_boxes[k, 3:6] = torch.clamp(gt_boxes[k, 3:6], min=1e-3).log()  # log编码尺寸
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])  # 角度编码(cos)
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])  # 角度编码(sin)

        return heatmap, ret_boxes, inds, mask

    def _draw_gaussian_to_heatmap_voxels(self, heatmap, distances, radius, k=1):
        """
        在稀疏体素热图上绘制高斯分布。
        
        Args:
            heatmap: (num_voxels,) 热图向量
            distances: (num_voxels,) 距离平方
            radius: float, 高斯半径
            k: float, 高斯系数
        """
        # 添加数值稳定性
        radius = max(radius, 1.0)  # 确保radius至少为1
        diameter = 2 * radius + 1
        sigma = max(diameter / 6.0, 0.5)  # 确保sigma不会太小
        
        # 限制距离范围防止数值问题
        distances_clamped = torch.clamp(distances, min=0, max=1e6)
        gaussian = torch.exp(-distances_clamped / (2 * sigma * sigma + 1e-6))
        
        # 确保gaussian值在合理范围内
        gaussian = torch.clamp(gaussian, min=0, max=1)
        torch.max(heatmap, gaussian, out=heatmap)

    def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """分配所有batch和head的目标"""
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                
                # 过滤padding的gt_boxes (全0的box)
                valid_mask = (cur_gt_boxes[:, 3:6].abs().sum(dim=1) > 0)
                cur_gt_boxes = cur_gt_boxes[valid_mask]
                
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for idx2, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx2].clone()
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                # 确保num_voxels是Python int
                cur_num_voxels = int(num_voxels[bs_idx].item()) if torch.is_tensor(num_voxels[bs_idx]) else int(num_voxels[bs_idx])
                
                heatmap, ret_boxes, inds, mask_out = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), 
                    gt_boxes=gt_boxes_single_head,
                    num_voxels=cur_num_voxels, 
                    spatial_indices=spatial_indices[bs_idx],
                    spatial_shape=spatial_shape,
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask_out.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            # 合并heatmaps: 每个batch的heatmap形状是 (num_classes, num_voxels_i)
            # cat后变成 (num_classes, total_voxels), permute后变成 (total_voxels, num_classes)
            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)

        return ret_dict

    def sigmoid(self, x):
        """安全的sigmoid，防止数值溢出导致NaN"""
        # 先处理输入中的NaN和Inf
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, min=-20, max=20)  # 防止极端值
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        """计算损失"""
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        tb_dict = {}
        # 使用第一个head的预测来创建与计算图连接的loss
        loss = None
        
        # 获取每个HEAD的权重（用于类别不平衡，如小目标Diver需要更高权重）
        # 配置示例: HEAD_WEIGHTS: [1.0, 2.0] 表示第二个HEAD（Diver）权重是2倍
        head_weights = self.model_cfg.LOSS_CONFIG.get('HEAD_WEIGHTS', None)
        if head_weights is None:
            head_weights = [1.0] * len(pred_dicts)

        for idx, pred_dict in enumerate(pred_dicts):
            # 获取当前HEAD的权重
            cur_head_weight = head_weights[idx] if idx < len(head_weights) else 1.0
            
            # 安全的sigmoid转换
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            
            # 静默处理NaN（移除debug输出）
            if torch.isnan(pred_dict['hm']).any():
                pred_dict['hm'] = torch.where(torch.isnan(pred_dict['hm']), 
                                             torch.full_like(pred_dict['hm'], 0.1), 
                                             pred_dict['hm'])
            
            # 分类损失 (Focal Loss)
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss = hm_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            # 回归损失 (L1 Loss)
            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            
            # 静默处理NaN
            if torch.isnan(pred_boxes).any():
                pred_boxes = torch.where(torch.isnan(pred_boxes), 
                                        torch.zeros_like(pred_boxes), 
                                        pred_boxes)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            
            # 计算loc_loss
            code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
            loc_loss = (reg_loss * reg_loss.new_tensor(code_weights)).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            
            # NaN安全处理 - 用与计算图连接的值替代
            if torch.isnan(hm_loss) or torch.isinf(hm_loss):
                hm_loss = (pred_dict['hm'] * 0).sum()
            if torch.isnan(loc_loss) or torch.isinf(loc_loss):
                loc_loss = (pred_boxes * 0).sum()
            
            # IoU回归损失（v4新增）
            iou_loss = pred_boxes.new_tensor(0.0)
            if self.use_iou and 'iou' in pred_dict:
                iou_loss = self._compute_iou_loss(
                    iou_preds=pred_dict['iou'],
                    pred_boxes_all=pred_boxes,
                    target_boxes=target_boxes,
                    masks=target_dicts['masks'][idx],
                    inds=target_dicts['inds'][idx],
                    gt_boxes_list=target_dicts['gt_boxes'][idx],
                    batch_index=batch_index
                )
                iou_loss = iou_loss * self.iou_loss_weight
            
            # 应用HEAD权重（为小目标类别增加权重）
            head_loss = (hm_loss + loc_loss + iou_loss) * cur_head_weight
            
            # 记录tensorboard值
            tb_dict[f'hm_loss_head_{idx}'] = hm_loss.detach().item() if not torch.isnan(hm_loss.detach()) else 0.0
            tb_dict[f'loc_loss_head_{idx}'] = loc_loss.detach().item() if not torch.isnan(loc_loss.detach()) else 0.0
            if self.use_iou:
                tb_dict[f'iou_loss_head_{idx}'] = iou_loss.detach().item() if not torch.isnan(iou_loss.detach()) else 0.0
            
            if loss is None:
                loss = head_loss
            else:
                loss = loss + head_loss

        # 确保loss有梯度
        if not loss.requires_grad:
            loss = loss.clone().requires_grad_(True)
            
        tb_dict['rpn_loss'] = loss.detach().item() if not torch.isnan(loss) else 0.0
        return loss, tb_dict

    def _compute_iou_loss(self, iou_preds, pred_boxes_all, target_boxes, masks, inds, gt_boxes_list, batch_index):
        """
        计算IoU回归损失（v4新增）。
        
        对每个正样本体素：
        1. 根据预测的回归值构建pred_box
        2. 与对应GT box计算3D IoU
        3. 用L1 loss监督IoU预测分支
        
        Args:
            iou_preds: (N, 1) 所有体素的IoU预测
            pred_boxes_all: (N, dim) 所有体素的回归预测（已cat）
            target_boxes: (batch, max_objs, dim) 回归目标
            masks: (batch, max_objs) 有效mask
            inds: (batch, max_objs) 目标体素索引
            gt_boxes_list: list of (Mi, 7) 每个batch的GT boxes
            batch_index: (N,) 体素batch索引
        """
        if masks.sum() == 0:
            return iou_preds.new_tensor(0.0)
        
        if iou3d_nms_utils is None:
            return iou_preds.new_tensor(0.0)
        
        batch_size = masks.shape[0]
        all_iou_preds = []
        all_iou_targets = []
        
        for bs_idx in range(batch_size):
            cur_mask = masks[bs_idx].bool()  # (max_objs,)
            if cur_mask.sum() == 0:
                continue
            
            batch_inds = batch_index == bs_idx
            cur_iou_preds = iou_preds[batch_inds]  # (M, 1)
            cur_pred_boxes_all = pred_boxes_all[batch_inds]  # (M, dim)
            
            if cur_iou_preds.shape[0] == 0:
                continue
            
            # 获取正样本体素的预测
            cur_inds = inds[bs_idx][cur_mask].clamp(0, cur_iou_preds.shape[0] - 1)
            selected_iou_preds = cur_iou_preds[cur_inds].squeeze(-1)  # (K,)
            selected_pred_regress = cur_pred_boxes_all[cur_inds]  # (K, dim)
            
            # 获取spatial_indices来解码pred box
            cur_voxel_indices = self.forward_ret_dict['voxel_indices'][batch_inds]
            cur_spatial = cur_voxel_indices[:, 1:]  # [y, x]
            selected_spatial = cur_spatial[cur_inds]
            
            # 解码pred box: 与_get_predicted_boxes一致
            center_offset = selected_pred_regress[:, 0:2]  # [offset_y, offset_x]
            center_z = selected_pred_regress[:, 2:3]
            dim = torch.exp(torch.clamp(selected_pred_regress[:, 3:6], min=-5, max=5))
            rot_cos = selected_pred_regress[:, 6:7]
            rot_sin = selected_pred_regress[:, 7:8]
            angle = torch.atan2(rot_sin, rot_cos)
            
            xs = (selected_spatial[:, 1:2].float() + center_offset[:, 1:2]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            ys = (selected_spatial[:, 0:1].float() + center_offset[:, 0:1]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            
            pred_boxes_decoded = torch.cat([xs, ys, center_z, dim, angle], dim=-1)  # (K, 7)
            
            # 获取对应GT boxes
            cur_target_boxes = target_boxes[bs_idx][cur_mask]  # (K, dim)
            # 解码GT boxes
            gt_center_offset = cur_target_boxes[:, 0:2]
            gt_z = cur_target_boxes[:, 2:3]
            gt_dim = torch.exp(torch.clamp(cur_target_boxes[:, 3:6], min=-5, max=5))
            gt_rot_cos = cur_target_boxes[:, 6:7]
            gt_rot_sin = cur_target_boxes[:, 7:8]
            gt_angle = torch.atan2(gt_rot_sin, gt_rot_cos)
            
            gt_xs = (selected_spatial[:, 1:2].float() + gt_center_offset[:, 1:2]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            gt_ys = (selected_spatial[:, 0:1].float() + gt_center_offset[:, 0:1]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            
            gt_boxes_decoded = torch.cat([gt_xs, gt_ys, gt_z, gt_dim, gt_angle], dim=-1)  # (K, 7)
            
            # 计算paired IoU
            try:
                iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(
                    pred_boxes_decoded.detach()[:, 0:7], 
                    gt_boxes_decoded[:, 0:7]
                )
                iou_target = iou_target * 2 - 1  # [0,1] → [-1,1] 映射
                iou_target = torch.clamp(iou_target, min=-1, max=1)
                
                all_iou_preds.append(selected_iou_preds)
                all_iou_targets.append(iou_target)
            except Exception:
                continue
        
        if len(all_iou_preds) == 0:
            return iou_preds.new_tensor(0.0)
        
        all_iou_preds = torch.cat(all_iou_preds)
        all_iou_targets = torch.cat(all_iou_targets)
        
        # 过滤NaN
        valid = ~(torch.isnan(all_iou_preds) | torch.isnan(all_iou_targets))
        if valid.sum() == 0:
            return iou_preds.new_tensor(0.0)
        
        loss = F.l1_loss(all_iou_preds[valid], all_iou_targets[valid], reduction='sum')
        loss = loss / torch.clamp(valid.float().sum(), min=1.0)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return iou_preds.new_tensor(0.0)
        
        return loss

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        """从预测构建3D框"""
        center = pred_dict['center']  # (N, 2) [offset_y, offset_x]
        center_z = pred_dict['center_z']
        # 防止dim回归发散导致exp溢出
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        
        # 注意：spatial_indices是[y, x], center是[offset_y, offset_x]
        # xs = (x + offset_x) * ...
        # ys = (y + offset_y) * ...
        xs = (spatial_indices[:, 1:2].float() + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1].float() + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        pred_box = torch.cat([xs, ys, center_z, dim, angle], dim=-1)
        return pred_box

    def _get_voxel_infos(self, x):
        """提取稀疏tensor的体素信息"""
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, 1:])  # [y, x]
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def _suppress_cross_class_duplicates(self, boxes, scores, labels, post_process_cfg):
        """
        跨类别冲突抑制：若不同类别在同一位置高度重叠，只保留高分框。

        说明：
        - 当前解码采用per-class TopK + per-class NMS，会允许同位异类框共存；
        - 本步骤作为额外约束，不改变类内NMS，仅处理异类冲突。
        """
        enable = post_process_cfg.get('ENABLE_CROSS_CLASS_SUPPRESS', True)
        if (not enable) or boxes.shape[0] <= 1:
            return boxes, scores, labels

        center_dist_thresh = post_process_cfg.get('CROSS_CLASS_CENTER_DIST_THRESH', 1.0)
        bev_iou_thresh = post_process_cfg.get('CROSS_CLASS_BEV_IOU_THRESH', 0.55)
        size_similarity_thresh = post_process_cfg.get('CROSS_CLASS_SIZE_SIMILARITY_THRESH', 0.6)

        order = torch.argsort(scores, descending=True)
        keep_mask = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        iou_matrix = None
        if iou3d_nms_utils is not None and bev_iou_thresh > 0:
            try:
                iou_matrix = iou3d_nms_utils.boxes_iou_bev(boxes[:, 0:7], boxes[:, 0:7])
            except Exception:
                iou_matrix = None

        for i in range(order.shape[0]):
            idx_i = order[i]
            if not keep_mask[idx_i]:
                continue

            for j in range(i + 1, order.shape[0]):
                idx_j = order[j]
                if (not keep_mask[idx_j]) or (labels[idx_i] == labels[idx_j]):
                    continue

                center_dist = torch.norm(boxes[idx_i, 0:2] - boxes[idx_j, 0:2], p=2)
                center_close = center_dist <= center_dist_thresh

                bev_overlap = False
                if iou_matrix is not None:
                    bev_overlap = iou_matrix[idx_i, idx_j] >= bev_iou_thresh

                # 体积相似度: min(vol_i, vol_j) / max(vol_i, vol_j)
                vol_i = torch.clamp(boxes[idx_i, 3] * boxes[idx_i, 4] * boxes[idx_i, 5], min=1e-4)
                vol_j = torch.clamp(boxes[idx_j, 3] * boxes[idx_j, 4] * boxes[idx_j, 5], min=1e-4)
                size_similarity = torch.min(vol_i, vol_j) / torch.max(vol_i, vol_j)
                size_similar = size_similarity >= size_similarity_thresh

                # 更保守：必须“中心接近 + BEV高重叠 + 尺寸相似”才认为是跨类重复框
                conflict = center_close and bev_overlap and size_similar

                if conflict:
                    keep_mask[idx_j] = False

        return boxes[keep_mask], scores[keep_mask], labels[keep_mask]

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        """生成预测框 - 使用per-class NMS防止跨类别抑制"""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            
            # IoU-aware score rectification (v4新增)
            # final_score = hm_score^(1-w) * iou_pred^w，w为rectify权重
            if self.use_iou and 'iou' in pred_dict:
                batch_iou = torch.clamp((pred_dict['iou'].squeeze(-1) + 1) / 2, min=0, max=1)  # [-1,1] → [0,1]
                w = self.iou_rectify_weight
                batch_hm = torch.pow(batch_hm, 1 - w) * torch.pow(batch_iou.unsqueeze(-1).expand_as(batch_hm), w)
            
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            # 防止exp溢出
            batch_dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)

            final_pred_dicts = self._decode_bbox_from_voxels(
                batch_size=batch_size, indices=voxel_indices,
                obj=batch_hm,
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z,
                dim=batch_dim,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                
                # ===== 关键修改：per-class NMS（替代class_agnostic_nms）=====
                # class_agnostic_nms会导致高分Box抑制空间重叠的低分Diver
                if final_dict['pred_boxes'].shape[0] > 0:
                    all_boxes, all_scores, all_labels = [], [], []
                    unique_labels = final_dict['pred_labels'].unique()
                    
                    for cls_label in unique_labels:
                        cls_mask = final_dict['pred_labels'] == cls_label
                        cls_boxes = final_dict['pred_boxes'][cls_mask]
                        cls_scores = final_dict['pred_scores'][cls_mask]
                        
                        if cls_boxes.shape[0] > 0:
                            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                                box_scores=cls_scores, box_preds=cls_boxes,
                                nms_config=post_process_cfg.NMS_CONFIG,
                                score_thresh=None
                            )
                            all_boxes.append(cls_boxes[selected])
                            all_scores.append(selected_scores)
                            all_labels.append(torch.full((selected.shape[0],), cls_label.item(), 
                                                        device=cls_boxes.device, dtype=final_dict['pred_labels'].dtype))
                    
                    if len(all_boxes) > 0:
                        final_dict['pred_boxes'] = torch.cat(all_boxes, dim=0)
                        final_dict['pred_scores'] = torch.cat(all_scores, dim=0)
                        final_dict['pred_labels'] = torch.cat(all_labels, dim=0)

                        # 跨类别冲突抑制：同位异类仅保留高分框
                        final_dict['pred_boxes'], final_dict['pred_scores'], final_dict['pred_labels'] = \
                            self._suppress_cross_class_duplicates(
                                final_dict['pred_boxes'],
                                final_dict['pred_scores'],
                                final_dict['pred_labels'],
                                post_process_cfg
                            )
                    else:
                        final_dict['pred_boxes'] = torch.zeros((0, 7), device='cuda')
                        final_dict['pred_scores'] = torch.zeros((0,), device='cuda')
                        final_dict['pred_labels'] = torch.zeros((0,), device='cuda', dtype=torch.long)

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            if len(ret_dict[k]['pred_boxes']) > 0:
                ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
                ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
                ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1
            else:
                ret_dict[k]['pred_boxes'] = torch.zeros((0, 7), device='cuda')
                ret_dict[k]['pred_scores'] = torch.zeros((0,), device='cuda')
                ret_dict[k]['pred_labels'] = torch.zeros((0,), device='cuda', dtype=torch.long)

        return ret_dict

    def _decode_bbox_from_voxels(self, batch_size, indices, obj, rot_cos, rot_sin, 
                                  center, center_z, dim, K=100, score_thresh=None, 
                                  post_center_limit_range=None):
        """
        从稀疏体素预测解码3D边界框。
        
        关键修复v2：使用 **真正的per-class TopK** —— 每个类别独立选取TopK，
        然后合并结果。这保证每个类别都有独立的预测名额，不会被多数类完全压制。
        
        v1的全局flatten TopK仍然有隐性竞争问题（Box高分体素多→占满名额）。
        
        Args:
            batch_size: int
            indices: (N, 3) [batch, y, x]
            obj: (N, num_classes) 热图预测（已sigmoid）
            rot_cos, rot_sin: (N, 1) 角度预测
            center: (N, 2) XY偏移
            center_z: (N, 1) Z坐标
            dim: (N, 3) 尺寸
            K: int, 每个batch每个类别最多保留的目标数
            score_thresh: float, 分数阈值
            post_center_limit_range: (6,) 坐标范围限制
        """
        batch_idx = indices[:, 0]
        spatial_indices = indices[:, 1:]  # [y, x]
        num_classes = obj.shape[1]
        
        # 每个类别的TopK数量（总K平均分配，保证每个类都有机会）
        K_per_class = max(K // num_classes, 10)  # 至少每类10个
        
        ret_pred_dicts = []
        
        for bs_idx in range(batch_size):
            batch_mask = batch_idx == bs_idx
            cur_obj = obj[batch_mask]  # (M, num_classes)
            cur_center = center[batch_mask]
            cur_center_z = center_z[batch_mask]
            cur_dim = dim[batch_mask]
            cur_rot_cos = rot_cos[batch_mask]
            cur_rot_sin = rot_sin[batch_mask]
            cur_spatial = spatial_indices[batch_mask]
            
            M = cur_obj.shape[0]
            
            if M == 0:
                ret_pred_dicts.append({
                    'pred_boxes': torch.zeros((0, 7), device=obj.device),
                    'pred_scores': torch.zeros((0,), device=obj.device),
                    'pred_labels': torch.zeros((0,), device=obj.device, dtype=torch.long)
                })
                continue
            
            # ========== 真正的 per-class TopK ==========
            # 每个类别独立选取最高分的体素，保证少数类有独立的预测名额
            all_scores = []
            all_voxel_inds = []
            all_classes = []
            
            for c in range(num_classes):
                cls_scores = cur_obj[:, c]  # (M,)
                k_c = min(K_per_class, M)
                topk_scores_c, topk_inds_c = torch.topk(cls_scores, k_c)
                
                # per-class分数过滤
                if score_thresh is not None:
                    thresh_mask = topk_scores_c > score_thresh
                    topk_scores_c = topk_scores_c[thresh_mask]
                    topk_inds_c = topk_inds_c[thresh_mask]
                
                if topk_scores_c.shape[0] > 0:
                    all_scores.append(topk_scores_c)
                    all_voxel_inds.append(topk_inds_c)
                    all_classes.append(torch.full_like(topk_inds_c, c, dtype=torch.long))
            
            if len(all_scores) == 0:
                ret_pred_dicts.append({
                    'pred_boxes': torch.zeros((0, 7), device=obj.device),
                    'pred_scores': torch.zeros((0,), device=obj.device),
                    'pred_labels': torch.zeros((0,), device=obj.device, dtype=torch.long)
                })
                continue
            
            topk_scores = torch.cat(all_scores, dim=0)
            topk_voxel_inds = torch.cat(all_voxel_inds, dim=0)
            topk_classes = torch.cat(all_classes, dim=0)
            
            # 获取选中的预测（用体素索引）
            sel_center = cur_center[topk_voxel_inds]
            sel_center_z = cur_center_z[topk_voxel_inds]
            sel_dim = cur_dim[topk_voxel_inds]
            sel_rot_cos = cur_rot_cos[topk_voxel_inds]
            sel_rot_sin = cur_rot_sin[topk_voxel_inds]
            sel_spatial = cur_spatial[topk_voxel_inds]
            sel_classes = topk_classes
            
            # 解码坐标
            angle = torch.atan2(sel_rot_sin, sel_rot_cos)
            xs = (sel_spatial[:, 1:2].float() + sel_center[:, 1:2]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            ys = (sel_spatial[:, 0:1].float() + sel_center[:, 0:1]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            
            pred_boxes = torch.cat([xs, ys, sel_center_z, sel_dim, angle], dim=-1)
            
            # 坐标范围过滤
            if post_center_limit_range is not None:
                center_mask = (pred_boxes[:, :3] >= post_center_limit_range[:3]).all(dim=1)
                center_mask &= (pred_boxes[:, :3] <= post_center_limit_range[3:]).all(dim=1)
                pred_boxes = pred_boxes[center_mask]
                topk_scores = topk_scores[center_mask]
                sel_classes = sel_classes[center_mask]
            
            ret_pred_dicts.append({
                'pred_boxes': pred_boxes,
                'pred_scores': topk_scores,
                'pred_labels': sel_classes
            })
        
        return ret_pred_dicts

    def forward(self, data_dict):
        """前向传播"""
        x = data_dict['encoded_spconv_tensor']

        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['voxel_indices'] = voxel_indices

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'],
                pred_dicts, voxel_indices, spatial_shape
            )
            data_dict['final_box_dicts'] = pred_dicts

        return data_dict


# ============= 稀疏损失函数 ==============

def neg_loss_sparse(pred, gt):
    """
    稀疏Focal Loss (Modified CornerNet Loss)。
    
    针对稀疏检测做了以下修改：
    1. 当num_pos=0时，用体素总数归一化neg_loss（避免loss过大）
    2. 增强数值稳定性
    
    Args:
        pred: (N, num_classes) 预测概率 (已经过sigmoid)
        gt: (N, num_classes) 高斯热图目标
        
    Returns:
        loss: scalar
    """
    # 边界情况检查
    if pred.numel() == 0 or gt.numel() == 0:
        return pred.new_tensor(0.0)
    
    # 确保pred在安全范围内
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    
    # 处理可能的NaN
    pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0.1), pred)
    gt = torch.where(torch.isnan(gt), torch.zeros_like(gt), gt)
    
    # 关键修复：使用>=0.999而不是==1判断正样本
    # 浮点精度问题可能导致1.0存储为0.9999...，导致正样本丢失
    pos_inds = gt.ge(0.999).float()
    neg_inds = gt.lt(0.999).float()
    neg_weights = torch.pow(1 - gt + 1e-6, 4)

    # Focal Loss计算 - 使用安全的log操作避免FP16溢出
    # log(x)在x->0时->-inf，clamp保证数值安全
    safe_log_pred = torch.log(torch.clamp(pred, min=1e-6))
    safe_log_1_pred = torch.log(torch.clamp(1 - pred, min=1e-6))
    
    pos_loss = safe_log_pred * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = safe_log_1_pred * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()
    
    # 使用clamp防止中间结果溢出
    pos_loss_sum = torch.clamp(pos_loss.sum(), min=-1e6, max=1e6)
    neg_loss_sum = torch.clamp(neg_loss.sum(), min=-1e6, max=1e6)

    if num_pos == 0:
        # 当没有正样本时，用负样本数量归一化
        if num_neg > 0:
            loss = -neg_loss_sum / torch.clamp(num_neg, min=1.0)
        else:
            # 返回一个与pred连接的0 tensor以保持计算图
            loss = (pred * 0).sum()
    else:
        loss = -(pos_loss_sum + neg_loss_sum) / torch.clamp(num_pos, min=1.0)
    
    # 最终NaN和异常值检查 - 返回与计算图连接的0
    if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
        return (pred * 0).sum()
    
    return loss


class FocalLossSparse(nn.Module):
    """稀疏Focal Loss模块"""
    def __init__(self):
        super(FocalLossSparse, self).__init__()
        self.neg_loss = neg_loss_sparse

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLossSparse(nn.Module):
    """稀疏回归损失模块 (L1 Loss)"""
    def __init__(self):
        super(RegLossSparse, self).__init__()

    def forward(self, output, mask, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N, dim) 所有体素的预测
            mask: (batch, max_objects) 有效目标mask
            ind: (batch, max_objects) 目标体素索引
            target: (batch, max_objects, dim) 目标值
            batch_index: (N,) 每个体素属于哪个batch
            
        Returns:
            loss: (dim,) 各维度的损失
        """
        batch_size = mask.shape[0]
        dim = target.shape[-1]
        
        # 边界情况：无有效目标
        num = mask.float().sum()
        if num == 0:
            return output.new_zeros(dim)
        
        pred = []
        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            cur_output = output[batch_inds]
            
            # 边界情况：当前batch无体素
            if cur_output.shape[0] == 0:
                # 创建零预测
                pred.append(output.new_zeros(ind.shape[1], output.shape[-1]))
                continue
            
            # 防止索引越界
            cur_ind = ind[bs_idx].clamp(0, cur_output.shape[0] - 1)
            pred.append(cur_output[cur_ind])
        
        pred = torch.stack(pred)  # (batch, max_objects, dim)

        # L1 Loss with mask
        mask_expanded = mask.unsqueeze(2).expand_as(target).float()
        
        # 过滤NaN
        isnotnan_target = (~torch.isnan(target)).float()
        isnotnan_pred = (~torch.isnan(pred)).float()
        valid_mask = mask_expanded * isnotnan_target * isnotnan_pred
        
        # 将NaN替换为0进行计算
        pred_safe = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        target_safe = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        
        pred_masked = pred_safe * valid_mask
        target_masked = target_safe * valid_mask

        loss = torch.abs(pred_masked - target_masked)
        loss = loss.transpose(2, 0)  # (dim, batch, max_objects)
        loss = torch.sum(loss, dim=2)  # (dim, batch)
        loss = torch.sum(loss, dim=1)  # (dim,)
        loss = loss / torch.clamp_min(num, min=1.0)
        
        # 最终NaN检查
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return loss
