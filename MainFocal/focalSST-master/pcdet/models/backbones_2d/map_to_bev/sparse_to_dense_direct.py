import torch
import torch.nn as nn


class SparseToDenseDirect(nn.Module):
    """
    Directly convert sparse 3D features to dense 2D BEV features
    without calling dense() on the entire volume. This is much more
    memory-efficient for high-resolution sparse tensors.
    
    Compatible with both:
    - encoded_spconv_tensor (from FocalSparseEncoder)
    - voxel_features + voxel_coords (from DSVT)
    """
    def __init__(self, model_cfg, grid_size=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.grid_size = grid_size  # [X, Y, Z]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor (optional, from FocalSparseEncoder)
                OR
                voxel_features: [N, C] (from DSVT)
                voxel_coords: [N, 4] [batch_idx, z, y, x] (from DSVT)
        Returns:
            batch_dict:
                spatial_features: [N, C, H, W] - BEV features
        """
        import spconv.pytorch as spconv
        
        # Check which input format we have
        if 'encoded_spconv_tensor' in batch_dict:
            # Original path: spconv tensor from FocalSparseEncoder
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_shape = encoded_spconv_tensor.spatial_shape  # [Z, Y, X]
            batch_size = batch_dict['batch_size']
            
            # Get features and indices from sparse tensor
            features = encoded_spconv_tensor.features  # [num_active, C]
            indices = encoded_spconv_tensor.indices    # [num_active, 4] where 4 is [batch_idx, z, y, x]
            
        elif 'voxel_features' in batch_dict and 'voxel_coords' in batch_dict:
            # DSVT path: voxel features + coordinates
            features = batch_dict['voxel_features']  # [N, C]
            indices = batch_dict['voxel_coords']     # [N, 4] [batch_idx, z, y, x]
            batch_size = batch_dict['batch_size']
            
            # Get spatial shape from grid_size
            if self.grid_size is not None:
                # grid_size is [X, Y, Z], spatial_shape needs [Z, Y, X]
                spatial_shape = [self.grid_size[2], self.grid_size[1], self.grid_size[0]]
            else:
                # Infer from coordinates
                z_max = indices[:, 1].max().item() + 1
                y_max = indices[:, 2].max().item() + 1
                x_max = indices[:, 3].max().item() + 1
                spatial_shape = [z_max, y_max, x_max]
        else:
            raise ValueError("batch_dict must contain either 'encoded_spconv_tensor' or 'voxel_features'+'voxel_coords'")
        
        # Create output tensor [batch, C, Y, X] initialized with zeros
        # We ignore Z dimension by taking max pooling implicitly
        C = features.shape[1]
        Y, X = spatial_shape[1], spatial_shape[2]  # spatial_shape is [Z, Y, X]
        
        spatial_features = torch.zeros(
            (batch_size, C, Y, X),
            dtype=features.dtype,
            device=features.device
        )
        
        # Fill in the BEV features
        # For each active voxel, we place its features at the (y, x) location
        # If multiple voxels exist at the same (y, x), we use max pooling
        batch_idx = indices[:, 0].long()  # batch index
        # z_idx = indices[:, 1].long()    # z index (height) - we ignore this
        y_idx = indices[:, 2].long()      # y index
        x_idx = indices[:, 3].long()      # x index
        
        # Use scatter_reduce with 'amax' for max pooling over height
        # or 'mean' for average pooling
        for b in range(batch_size):
            mask = (batch_idx == b)
            if mask.sum() == 0:
                continue
            
            b_features = features[mask]  # [num_voxels_in_batch, C]
            b_y = y_idx[mask]
            b_x = x_idx[mask]
            
            # For each (y, x) location, take the max over all z values
            # Use scatter with reduction='max' to avoid inplace modification issues
            unique_yx = torch.unique(torch.stack([b_y, b_x], dim=1), dim=0)
            for yx in unique_yx:
                y, x = yx[0].item(), yx[1].item()
                # Find all features at this (y, x) location
                mask_yx = (b_y == y) & (b_x == x)
                features_at_yx = b_features[mask_yx]
                # Take max over all features at this location
                spatial_features[b, :, y, x] = features_at_yx.max(dim=0)[0]
        
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict.get('encoded_spconv_tensor_stride', 1)
        
        return batch_dict
