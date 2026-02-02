# 训练失败诊断报告

## 问题总结

经过深入调试，确定了训练失败的**根本原因**是**模型架构配置问题**，而非数据问题。

## 详细分析

### 1. 已解决的问题

✅ **Numpy兼容性问题**: 降级numpy从2.0.2到1.26.4，解决了cumm/spconv的SIGFPE错误  
✅ **conv_imp通道数错误**: 从1改为27（26个kernel位置 + 1个中心voxel）  
✅ **Focal loss监督**: 只监督最后一个通道的voxel importance  
✅ **输出通道匹配**: conv_focal从64改为128以匹配DSVT输入需求  
✅ **数据路径配置**: 修正了配置文件中的路径错误  

### 2. 核心问题：架构不兼容

**问题描述**:
- FocalSST + DSVT输出: 稀疏3D特征 `[N, 128, 175, 200, 500]` (sparse tensor)
- HeightCompression: 将高度维度压平 `[N, 128*175, 200, 500]` = `[N, 22400, 200, 500]`
- BEV Backbone: 期望接收合理数量的输入通道（如128-512），但实际收到22400通道

**导致结果**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.10 GiB
```

一个3x3卷积层处理22400通道的feature map需要巨大的内存：
- 输入: [1, 22400, 200, 500] 
- 第一层Conv2d(22400, 128, 3x3)的权重: 22400 * 128 * 3 * 3 = 25,804,800 参数
- 激活值占用约75GB显存

### 3. 为什么会这样？

这是将两个不同设计理念的模块强行组合导致的:

1. **DSVT**: 设计用于处理**稀疏3D点云**，输出是稀疏tensor
2. **标准HeightCompression**: 设计用于VoxelNet/SECOND等模型，这些模型的3D backbone已经通过下采样减小了spatial_shape
3. **冲突**: DSVT保持了完整的空间分辨率[175, 200, 500]，导致压平后通道数爆炸

## 确定性结论

**这不是数据问题，而是模型架构设计问题。**

当前的FocalSST+DSVT+HeightCompression组合在此数据配置下**无法直接工作**，需要架构层面的修改。

## 解决方案

### 方案1：修改HeightCompression（推荐）

使用learned或max pooling压缩高度维度到固定通道数：

```python
class SparseHeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES  # 例如256
        
        # 添加一个1x1卷积来压缩通道
        self.compress_conv = nn.Sequential(
            nn.Conv2d(model_cfg.INPUT_FEATURES, self.num_bev_features, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU()
        )
    
    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        
        # 方法1: Max pooling along height
        spatial_features = spatial_features.max(dim=2)[0]  # [N, C, H, W]
        
        # 方法2: Mean pooling
        # spatial_features = spatial_features.mean(dim=2)
        
        # 方法3: Learned projection
        # spatial_features = spatial_features.view(N, C*D, H, W)
        # spatial_features = self.compress_conv(spatial_features)
        
        batch_dict['spatial_features'] = spatial_features
        return batch_dict
```

### 方案2：在3D阶段下采样

修改focal_sparse_encoder或DSVT，增加下采样层：

```python
# 在DSVT配置中添加downsample_stride
INPUT_LAYER:
    sparse_shape: [500, 200, 175]
    downsample_stride: [1, 2, 4]  # Z方向下采样4倍: 175/4 ≈ 44
```

这样输出会变成 `[N, 128, 44, 200, 500]`，压平后是 `[N, 5632, 200, 500]`，显存需求下降到可接受范围。

### 方案3：使用不同的MAP_TO_BEV模块

OpenPCDet提供了`SparseHeightCompression`，可以尝试：

```yaml
MAP_TO_BEV:
    NAME: SparseHeightCompression
    NUM_BEV_FEATURES: 256
```

## 下一步建议

1. **立即可行**: 实现方案1的Max Pooling版本，这是最简单且有效的修改
2. **长期优化**: 研究原始FocalSST论文，看他们如何处理height compression
3. **备选方案**: 考虑使用更轻量的3D backbone（如VoxelNet）替代DSVT

