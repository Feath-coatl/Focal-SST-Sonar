"""
FocalSSTVoxelNeXt: 结合FocalDSVT特征提取与VoxelNeXt稀疏检测头的检测器。

架构流程:
1. VFE (SDAVFE): 体素特征提取
2. VoxelBackbone (FocalDSVT): Focal Sparse + DSVT Transformer
3. Map2BEV (FocalToVoxelNeXtBridge): 保height-sum的稀疏BEV映射
4. DenseHead (VoxelNeXtHeadSonar): 稀疏检测头，直接回归Z坐标

优势:
- 保留FocalDSVT强大的3D特征学习能力
- 使用VoxelNeXt的稀疏检测头，支持垂直方向目标检测
- 全稀疏推理，效率高
"""

from .detector3d_template import Detector3DTemplate


class FocalSSTVoxelNeXt(Detector3DTemplate):
    """
    结合FocalDSVT和VoxelNeXt优势的3D目标检测器。
    
    与FocalDSVT的主要区别:
    1. 使用FocalToVoxelNeXtBridge替代SparseToDenseDirect (sum pooling vs max pooling)
    2. 不使用BEVBackbone2D (BaseBEVBackbone)
    3. 使用VoxelNeXtHeadSonar替代CenterHead/SparseAnchorFreeHead
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        前向传播流程:
        1. VFE: 原始点云 -> 体素特征
        2. VoxelBackbone (FocalDSVT): 体素特征 -> 多尺度3D稀疏特征
        3. Map2BEV (FocalToVoxelNeXtBridge): 3D稀疏特征 -> 2D稀疏BEV特征
        4. DenseHead (VoxelNeXtHeadSonar): 2D稀疏特征 -> 3D边界框
        """
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        """计算训练损失"""
        disp_dict = {}
        
        # 获取检测头的损失
        loss_rpn, tb_dict = self.dense_head.get_loss()
        
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        """
        后处理: 从dense_head的预测中提取最终结果。
        
        VoxelNeXtHeadSonar已经在forward中生成了final_box_dicts，
        这里负责格式化输出和计算recall。
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        
        # 从dense_head获取预测结果
        final_box_dicts = batch_dict.get('final_box_dicts', None)
        
        pred_dicts = []
        recall_dict = {}
        
        for index in range(batch_size):
            if final_box_dicts is not None:
                pred_boxes = final_box_dicts[index]['pred_boxes']
                pred_scores = final_box_dicts[index]['pred_scores']
                pred_labels = final_box_dicts[index]['pred_labels']
            else:
                pred_boxes = batch_dict['batch_box_preds'][index]
                pred_scores = batch_dict['batch_cls_preds'][index]
                pred_labels = batch_dict['batch_pred_labels'][index]
            
            pred_dict = {
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels
            }
            pred_dicts.append(pred_dict)
            
            # 计算recall
            if 'gt_boxes' in batch_dict:
                recall_dict = self.generate_recall_record(
                    box_preds=pred_boxes,
                    recall_dict=recall_dict,
                    batch_index=index,
                    data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )
        
        return pred_dicts, recall_dict
