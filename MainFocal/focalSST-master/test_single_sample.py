#!/usr/bin/env python3
"""简单测试单个样本加载"""
import sys
import os
import numpy as np

# 启用浮点异常检测
np.seterr(all='raise')

# 手动模拟get_lidar的完整流程
data_path = '/root/autodl-tmp/code/Modelproject/MainFocal/focalSST-master/data/sonar/training/velodyne/a_0180077.txt'

print('='*80)
print(f'测试文件: {data_path}')
print('='*80)

# Step 1: 加载数据
print('\nStep 1: numpy.loadtxt...')
try:
    points = np.loadtxt(data_path, dtype=np.float32)
    print(f'  ✓ shape={points.shape}')
except Exception as e:
    print(f'  ❌ 失败: {e}')
    sys.exit(1)

# Step 2: 提取前4列
print('\nStep 2: 提取前4列...')
try:
    points = points[:, :4]
    print(f'  ✓ shape={points.shape}')
except Exception as e:
    print(f'  ❌ 失败: {e}')
    sys.exit(1)

# Step 3: 提取intensity
print('\nStep 3: 提取intensity...')
try:
    intensity = points[:, 3]
    print(f'  ✓ 范围: [{intensity.min():.2e}, {intensity.max():.2e}]')
except Exception as e:
    print(f'  ❌ 失败: {e}')
    sys.exit(1)

# Step 4: clip
print('\nStep 4: np.clip...')
try:
    INTENSITY_CLIP_MAX = 4.64e10
    intensity_clipped = np.clip(intensity, 0, INTENSITY_CLIP_MAX)
    print(f'  ✓ 范围: [{intensity_clipped.min():.2e}, {intensity_clipped.max():.2e}]')
except FloatingPointError as e:
    print(f'  ❌ FloatingPointError在clip: {e}')
    sys.exit(1)

# Step 5: log10(x + 1)
print('\nStep 5: np.log10(intensity + 1)...')
try:
    intensity_log = np.log10(intensity_clipped + 1)
    print(f'  ✓ 范围: [{intensity_log.min():.2f}, {intensity_log.max():.2f}]')
except FloatingPointError as e:
    print(f'  ❌ FloatingPointError在log10: {e}')
    sys.exit(1)

# Step 6: 除以11.0
print('\nStep 6: 除以11.0...')
try:
    LOG_NORM_DIVISOR = 11.0
    intensity_normalized = intensity_log / LOG_NORM_DIVISOR
    print(f'  ✓ 范围: [{intensity_normalized.min():.2f}, {intensity_normalized.max():.2f}]')
except FloatingPointError as e:
    print(f'  ❌ FloatingPointError在除法: {e}')
    sys.exit(1)

# Step 7: 赋值回points
print('\nStep 7: 赋值回points...')
try:
    points[:, 3] = intensity_normalized
    print(f'  ✓ 完成')
except Exception as e:
    print(f'  ❌ 失败: {e}')
    sys.exit(1)

# Step 8: 坐标对齐 (Y -> X, -X -> Y)
print('\nStep 8: 坐标对齐...')
try:
    aligned_points = np.zeros_like(points)
    aligned_points[:, 0] = points[:, 1]    # Y -> X
    aligned_points[:, 1] = -points[:, 0]   # -X -> Y
    aligned_points[:, 2] = points[:, 2]    # Z不变
    aligned_points[:, 3] = points[:, 3]    # Intensity不变
    print(f'  ✓ 完成, shape={aligned_points.shape}')
except Exception as e:
    print(f'  ❌ 失败: {e}')
    sys.exit(1)

# 最终检查
print('\n最终检查:')
print(f'  Points shape: {aligned_points.shape}')
print(f'  Points范围: [{aligned_points.min():.2f}, {aligned_points.max():.2f}]')
print(f'  Intensity范围: [{aligned_points[:, 3].min():.2f}, {aligned_points[:, 3].max():.2f}]')
print(f'  有NaN? {np.isnan(aligned_points).any()}')
print(f'  有Inf? {np.isinf(aligned_points).any()}')

print('\n' + '='*80)
print('✅ 所有操作成功！')
print('='*80)
