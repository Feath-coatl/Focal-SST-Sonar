import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        
        print(f"\n[DEBUG HeightCompression] Before dense():")
        print(f"  encoded_spconv_tensor.spatial_shape: {encoded_spconv_tensor.spatial_shape}")
        print(f"  encoded_spconv_tensor.features.shape: {encoded_spconv_tensor.features.shape}")
        print(f"  encoded_spconv_tensor.indices.shape: {encoded_spconv_tensor.indices.shape}")
        print(f"  encoded_spconv_tensor.batch_size: {encoded_spconv_tensor.batch_size}")
        
        spatial_features = encoded_spconv_tensor.dense()
        print(f"  After dense(), spatial_features.shape: {spatial_features.shape}")
        
        N, C, D, H, W = spatial_features.shape
        print(f"  Parsed: N={N}, C={C}, D={D}, H={H}, W={W}")
        
        spatial_features = spatial_features.view(N, C * D, H, W)
        print(f"  After view(N, C*D, H, W), spatial_features.shape: {spatial_features.shape}")
        print(f"  Expected: [batch={N}, channels={C*D}, H={H}, W={W}]\n")
        
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
