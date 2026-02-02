#!/usr/bin/env python3
"""
测试完整的数据加载流程，包括prepare_data中的所有操作
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from pcdet.datasets import SonarDataset
from easydict import EasyDict
import yaml

# 启用numpy的浮点异常检测
np.seterr(all='raise')

print('='*80)
print('测试完整数据加载流程（包括prepare_data）')
print('='*80)

# 加载配置
cfg = EasyDict(yaml.safe_load(open('tools/cfgs/sonar_models/focal_sst.yaml')))

# 创建dataset
dataset = SonarDataset(
    dataset_cfg=cfg.DATA_CONFIG, 
    class_names=cfg.CLASS_NAMES,
    training=True,
    root_path='data/sonar',
    logger=None
)

print(f'\nDataset大小: {len(dataset)}')
print(f'问题样本索引: 6651 (frame: a_0180077)')

# 尝试加载一个已知有问题的样本
test_indices = [6651]  # a_0180077对应的索引

for idx in test_indices:
    print(f'\n{"="*80}')
    print(f'测试索引 {idx} ...')
    
    try:
        # 第一步：只调用get_lidar
        print(f'  Step 1: 调用 get_lidar({idx})')
        points = dataset.get_lidar(idx)
        print(f'    ✓ get_lidar成功, shape={points.shape}')
        print(f'    ✓ Points范围: [{points.min():.2f}, {points.max():.2f}]')
        print(f'    ✓ Intensity范围: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]')
        
        # 检查是否有nan/inf
        if np.isnan(points).any():
            print(f'    ❌ 发现NaN值!')
            nan_mask = np.isnan(points)
            print(f'    NaN位置: {np.where(nan_mask)}')
        
        if np.isinf(points).any():
            print(f'    ❌ 发现Inf值!')
            inf_mask = np.isinf(points)
            print(f'    Inf位置: {np.where(inf_mask)}')
        
        # 第二步：调用完整的__getitem__
        print(f'\n  Step 2: 调用 dataset[{idx}] (包括prepare_data)')
        data_dict = dataset[idx]
        print(f'    ✓ __getitem__成功')
        print(f'    ✓ Keys: {list(data_dict.keys())}')
        
        if 'points' in data_dict:
            pts = data_dict['points']
            print(f'    ✓ Points shape: {pts.shape}')
            print(f'    ✓ Points范围: [{pts.min():.2f}, {pts.max():.2f}]')
            
            # 检查prepare_data后的points是否有问题
            if np.isnan(pts).any():
                print(f'    ❌ prepare_data后发现NaN值!')
            if np.isinf(pts).any():
                print(f'    ❌ prepare_data后发现Inf值!')
        
        print(f'\n  ✅ 样本 {idx} 加载完全正常!')
        
    except FloatingPointError as e:
        print(f'\n  ❌ FloatingPointError: {e}')
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f'\n  ❌ 其他错误: {e}')
        import traceback
        traceback.print_exc()

print(f'\n{"="*80}')
print('测试完成')
print('='*80)
