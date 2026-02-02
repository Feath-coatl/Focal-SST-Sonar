import torch.nn as nn


class SparseMaxHeightCompression(nn.Module):
    """
    Height compression using max pooling along the Z dimension.
    This is memory-efficient and works well with high-resolution 3D features.
    """
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor with shape [N, C, D, H, W] when densified
        Returns:
            batch_dict:
                spatial_features: [N, C, H, W] - max pooled along D dimension
        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        
        # Convert sparse tensor to dense: [N, C, D, H, W]
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        
        # Max pooling along the height (D/Z) dimension
        # This reduces [N, C, D, H, W] to [N, C, H, W]
        spatial_features = spatial_features.max(dim=2)[0]
        
        # 清理中间变量以释放内存
        del encoded_spconv_tensor
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict.get('encoded_spconv_tensor_stride', 1)
        
        return batch_dict
