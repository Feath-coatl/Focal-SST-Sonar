# -*- coding: utf-8 -*-
"""
数据集深度分析脚本 - 诊断Diver检测失败原因
在训练服务器上运行: python quick_analyze.py
"""
import json
import pickle
import os
import numpy as np
from collections import defaultdict

def analyze_dataset():
    print("=" * 70)
    print("SONAR数据集深度分析报告")
    print("=" * 70)
    
    # 1. 读取现有的pointcloud_statistics.json
    if os.path.exists('pointcloud_statistics.json'):
        data = json.load(open('pointcloud_statistics.json','r',encoding='utf-8'))
        print(f"\n[1] 原始点云统计 (pointcloud_statistics.json)")
        print(f"    总文件: {data['total_files']}")
        print(f"    总目标: {data['total_targets']}")
        for k,v in data['class_statistics'].items():
            pct = 100 * v['instance_count'] / data['total_targets']
            print(f"    类别{k} ({v['name']}): 实例数={v['instance_count']} ({pct:.1f}%), 平均点数={v['average_points']:.1f}")
    
    # 2. 分析pkl文件中的GT信息
    print(f"\n[2] GT标注统计 (sonar_infos_*.pkl)")
    
    all_stats = {}
    for split in ['train', 'val']:
        pkl_path = f'sonar_infos_{split}.pkl'
        if not os.path.exists(pkl_path):
            print(f"    警告: {pkl_path} 不存在")
            continue
        
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)
        
        class_stats = defaultdict(lambda: {
            'count': 0, 'frames': set(),
            'dx': [], 'dy': [], 'dz': [], 'z': [],
            'points': []
        })
        
        frames_with_no_gt = 0
        for info in infos:
            frame_id = info.get('point_cloud', {}).get('lidar_idx', 'unknown')
            
            if 'annos' not in info or info['annos'] is None:
                frames_with_no_gt += 1
                continue
            
            annos = info['annos']
            names = annos.get('name', [])
            
            if len(names) == 0:
                frames_with_no_gt += 1
                continue
            
            boxes = annos.get('gt_boxes_lidar', None)
            pts = annos.get('num_points_in_gt', None)
            
            for i, name in enumerate(names):
                if isinstance(name, np.ndarray):
                    name = str(name)
                class_stats[name]['count'] += 1
                class_stats[name]['frames'].add(frame_id)
                
                if boxes is not None and i < len(boxes):
                    class_stats[name]['dx'].append(boxes[i][3])
                    class_stats[name]['dy'].append(boxes[i][4])
                    class_stats[name]['dz'].append(boxes[i][5])
                    class_stats[name]['z'].append(boxes[i][2])
                
                if pts is not None and i < len(pts):
                    class_stats[name]['points'].append(pts[i])
        
        print(f"\n    --- {split.upper()} 集 ---")
        print(f"    总帧数: {len(infos)}, 无GT帧: {frames_with_no_gt}")
        
        total = sum(s['count'] for s in class_stats.values())
        
        for cls, stats in sorted(class_stats.items()):
            pct = 100 * stats['count'] / total if total > 0 else 0
            print(f"\n    [{cls}]")
            print(f"      实例数: {stats['count']} ({pct:.1f}%)")
            print(f"      出现帧数: {len(stats['frames'])}")
            
            if stats['dx']:
                dx = np.array(stats['dx'])
                dy = np.array(stats['dy'])
                dz = np.array(stats['dz'])
                z = np.array(stats['z'])
                vol = dx * dy * dz
                bev_area = dx * dy
                
                print(f"      尺寸统计:")
                print(f"        dx(前后): min={dx.min():.3f}, max={dx.max():.3f}, mean={dx.mean():.3f}, std={dx.std():.3f}")
                print(f"        dy(左右): min={dy.min():.3f}, max={dy.max():.3f}, mean={dy.mean():.3f}, std={dy.std():.3f}")
                print(f"        dz(高度): min={dz.min():.3f}, max={dz.max():.3f}, mean={dz.mean():.3f}, std={dz.std():.3f}")
                print(f"        体积:    min={vol.min():.3f}, max={vol.max():.3f}, mean={vol.mean():.3f}")
                print(f"        BEV面积: min={bev_area.min():.3f}, max={bev_area.max():.3f}, mean={bev_area.mean():.3f}")
                print(f"        Z高度:   min={z.min():.2f}, max={z.max():.2f}, mean={z.mean():.2f}")
                
                # 分析体素覆盖
                # VOXEL_SIZE: [0.25, 0.4, 0.25]
                vx, vy = 0.25, 0.4
                vox_x = dx / vx  # 目标在x方向覆盖的体素数
                vox_y = dy / vy  # 目标在y方向覆盖的体素数
                print(f"      体素覆盖(VOXEL_SIZE=[0.25,0.4,0.25]):")
                print(f"        X方向体素数: min={vox_x.min():.1f}, max={vox_x.max():.1f}, mean={vox_x.mean():.1f}")
                print(f"        Y方向体素数: min={vox_y.min():.1f}, max={vox_y.max():.1f}, mean={vox_y.mean():.1f}")
                
                # 统计小目标比例
                small_bev = np.sum(bev_area < 1.0)  # BEV面积<1m²
                tiny_bev = np.sum(bev_area < 0.5)   # BEV面积<0.5m²
                print(f"      小目标比例:")
                print(f"        BEV面积<1.0m²: {small_bev}/{len(bev_area)} ({100*small_bev/len(bev_area):.1f}%)")
                print(f"        BEV面积<0.5m²: {tiny_bev}/{len(bev_area)} ({100*tiny_bev/len(bev_area):.1f}%)")
        
        all_stats[split] = dict(class_stats)
    
    # 3. 诊断问题
    print(f"\n" + "=" * 70)
    print("问题诊断与建议")
    print("=" * 70)
    
    if 'train' in all_stats:
        train_stats = all_stats['train']
        
        # 检查类别不平衡
        if 'Box' in train_stats and 'Diver' in train_stats:
            box_count = train_stats['Box']['count']
            diver_count = train_stats['Diver']['count']
            
            print(f"\n[类别不平衡分析]")
            print(f"  Box实例数: {box_count}")
            print(f"  Diver实例数: {diver_count}")
            print(f"  比例: Box:Diver = {box_count/max(diver_count,1):.1f}:1")
            
            if diver_count == 0:
                print(f"  ⚠️ 严重问题: 训练集中没有Diver标注!")
                print(f"     可能原因: create_sonar_infos.py中点数过滤阈值(70点)导致Diver被过滤")
            elif diver_count < box_count * 0.1:
                print(f"  ⚠️ 严重不平衡: Diver样本过少")
            
            # 尺寸对比
            if train_stats['Box']['dx'] and train_stats['Diver']['dx']:
                box_bev = np.mean(np.array(train_stats['Box']['dx']) * np.array(train_stats['Box']['dy']))
                diver_bev = np.mean(np.array(train_stats['Diver']['dx']) * np.array(train_stats['Diver']['dy']))
                print(f"\n[尺寸对比]")
                print(f"  Box平均BEV面积: {box_bev:.3f}m²")
                print(f"  Diver平均BEV面积: {diver_bev:.3f}m²")
                print(f"  面积比: {box_bev/max(diver_bev,0.001):.1f}x")
                
                if diver_bev < 1.0:
                    print(f"  ⚠️ Diver目标较小(BEV<1m²)，可能需要:")
                    print(f"     1. 减小体素尺寸以获得更好的覆盖")
                    print(f"     2. 使用多尺度检测头")
                    print(f"     3. 为小目标单独设置检测头")
        
        elif 'Box' in train_stats and 'Diver' not in train_stats:
            print(f"\n⚠️ 严重问题: 训练集中完全没有Diver标注!")
            print(f"   这解释了为什么Diver检测率为0%")
            print(f"   请检查:")
            print(f"   1. create_sonar_infos.py中的点数过滤阈值(当前70点)")
            print(f"   2. 原始数据中Diver的点数分布")

if __name__ == '__main__':
    analyze_dataset()

