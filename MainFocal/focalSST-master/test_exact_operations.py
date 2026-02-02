#!/usr/bin/env python3
"""
精确重现 get_lidar 的每一步操作，找出触发浮点异常的确切步骤
"""
import numpy as np
from pathlib import Path

# 测试的文件
TEST_FILE = 'data/sonar/points/a_0180077.txt'

# 你的归一化参数
INTENSITY_CLIP_MAX = 4.64e10
LOG_NORM_DIVISOR = 11.0

print("=" * 80)
print("测试文件:", TEST_FILE)
print("=" * 80)

# 步骤1: 加载文件
print("\n【步骤1】使用 np.loadtxt 加载...")
try:
    points_all = np.loadtxt(TEST_FILE, dtype=np.float32)
    print(f"✓ 加载成功, shape: {points_all.shape}, dtype: {points_all.dtype}")
    print(f"  数据范围:")
    for i in range(min(5, points_all.shape[1])):
        print(f"    列{i}: [{points_all[:, i].min():.2e}, {points_all[:, i].max():.2e}]")
except Exception as e:
    print(f"❌ 步骤1失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤2: reshape (如果需要)
print("\n【步骤2】检查维度...")
if points_all.ndim == 1:
    print("  reshaping...")
    points_all = points_all.reshape(1, -1)
print(f"✓ shape: {points_all.shape}")

# 步骤3: 提取前4列
print("\n【步骤3】提取前4列...")
try:
    points = points_all[:, :4]
    print(f"✓ points shape: {points.shape}")
except Exception as e:
    print(f"❌ 步骤3失败: {e}")
    exit(1)

# 步骤4: 提取intensity
print("\n【步骤4】提取 intensity...")
try:
    intensity = points[:, 3]
    print(f"✓ intensity shape: {intensity.shape}")
    print(f"  范围: [{intensity.min():.2e}, {intensity.max():.2e}]")
    print(f"  均值: {intensity.mean():.2e}")
    print(f"  中位数: {np.median(intensity):.2e}")
    
    # 检查是否有异常值
    if np.any(np.isnan(intensity)):
        print(f"  ❌ 包含 NaN: {np.sum(np.isnan(intensity))}")
    if np.any(np.isinf(intensity)):
        print(f"  ❌ 包含 Inf: {np.sum(np.isinf(intensity))}")
    if np.any(intensity < 0):
        print(f"  ⚠️  包含负值: {np.sum(intensity < 0)}")
except Exception as e:
    print(f"❌ 步骤4失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤5: Clip操作
print("\n【步骤5】Clip intensity...")
try:
    intensity_clipped = np.clip(intensity, a_min=0, a_max=INTENSITY_CLIP_MAX)
    print(f"✓ Clip 成功")
    print(f"  Clip后范围: [{intensity_clipped.min():.2e}, {intensity_clipped.max():.2e}]")
    print(f"  Clip前后是否有变化: {not np.array_equal(intensity, intensity_clipped)}")
    print(f"  被clip的点数: {np.sum(intensity != intensity_clipped)}")
except Exception as e:
    print(f"❌ 步骤5失败 (这里可能触发浮点异常!): {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤6: 对数变换
print("\n【步骤6】对数变换 np.log10(intensity + 1)...")
try:
    print("  测试加法: intensity + 1...")
    intensity_plus_1 = intensity_clipped + 1
    print(f"  ✓ 加法成功, 范围: [{intensity_plus_1.min():.2e}, {intensity_plus_1.max():.2e}]")
    
    print("  测试 log10...")
    intensity_log = np.log10(intensity_plus_1)
    print(f"  ✓ log10 成功, 范围: [{intensity_log.min():.2f}, {intensity_log.max():.2f}]")
    
    if np.any(np.isnan(intensity_log)):
        print(f"  ❌ log10后包含 NaN: {np.sum(np.isnan(intensity_log))}")
    if np.any(np.isinf(intensity_log)):
        print(f"  ❌ log10后包含 Inf: {np.sum(np.isinf(intensity_log))}")
        
except Exception as e:
    print(f"❌ 步骤6失败 (这里可能触发浮点异常!): {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤7: 除法归一化
print("\n【步骤7】除法归一化 / LOG_NORM_DIVISOR...")
try:
    intensity_normalized = intensity_log / LOG_NORM_DIVISOR
    print(f"✓ 归一化成功")
    print(f"  归一化后范围: [{intensity_normalized.min():.2f}, {intensity_normalized.max():.2f}]")
    
    if np.any(np.isnan(intensity_normalized)):
        print(f"  ❌ 归一化后包含 NaN: {np.sum(np.isnan(intensity_normalized))}")
    if np.any(np.isinf(intensity_normalized)):
        print(f"  ❌ 归一化后包含 Inf: {np.sum(np.isinf(intensity_normalized))}")
        
except Exception as e:
    print(f"❌ 步骤7失败 (这里可能触发浮点异常!): {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤8: 赋值回points
print("\n【步骤8】赋值 points[:, 3] = intensity...")
try:
    points[:, 3] = intensity_normalized
    print(f"✓ 赋值成功")
except Exception as e:
    print(f"❌ 步骤8失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤9: 坐标对齐
print("\n【步骤9】坐标对齐...")
try:
    points_aligned = np.zeros_like(points)
    points_aligned[:, 0] = points[:, 1]   # Old Y -> New X
    points_aligned[:, 1] = -points[:, 0]  # -Old X -> New Y
    points_aligned[:, 2] = points[:, 2]   # Z保留
    points_aligned[:, 3] = points[:, 3]   # Intensity保留
    print(f"✓ 坐标对齐成功")
    print(f"  最终shape: {points_aligned.shape}")
    print(f"  最终强度范围: [{points_aligned[:, 3].min():.2f}, {points_aligned[:, 3].max():.2f}]")
    
    if np.any(np.isnan(points_aligned)):
        print(f"  ❌ 最终数据包含 NaN")
    if np.any(np.isinf(points_aligned)):
        print(f"  ❌ 最终数据包含 Inf")
except Exception as e:
    print(f"❌ 步骤9失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ 所有步骤成功完成！")
print("=" * 80)
print("\n如果此脚本运行成功，说明单独执行每个操作都没问题")
print("问题可能出在:")
print("1. 多进程/多线程环境下的数据竞争")
print("2. DataLoader的batch处理过程")
print("3. prepare_data中的数据增强操作")
print("\n下一步建议: 检查 prepare_data 方法中的操作")
