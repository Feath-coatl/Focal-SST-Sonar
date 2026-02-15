"""
Step 7: 参数调优 — Grid Search
在tracker+eval上快速遍历参数组合，找到最优配置。
每组参数仅需运行tracker+eval（无需推理），< 10秒/组。

搜索空间:
  - match_thresh: Box [2.0, 3.0, 4.0], Diver [1.5, 2.0, 3.0]
  - alpha: [0.1, 0.2, 0.3, 0.4, 0.5]
  - recovery_factor: [0.4, 0.6, 0.8]
  - fp_thresh: [0.2, 0.3, 0.4]
"""

import sys
import json
import pickle
import time
import itertools
from pathlib import Path
from copy import deepcopy


def setup_paths():
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'tools'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return project_root


def default_search_space():
    """默认搜索空间"""
    return {
        'match_thresh_box': [2.0, 2.5, 3.0],
        'match_thresh_diver': [1.0, 1.5, 2.0],
        'alpha': [0.3, 0.4, 0.5],
        'recovery_factor': [0.3, 0.4, 0.5],
        'fp_thresh': [0.3, 0.35, 0.4],
        'min_track_score': [0.4, 0.5, 0.6],
        'max_recovery_gap': [2, 3, 4],
    }


def fast_search_space():
    """精简搜索空间（用于快速验证）"""
    return {
        'match_thresh_box': [2.0, 2.5, 3.0],
        'match_thresh_diver': [1.0, 1.5, 2.0],
        'alpha': [0.3, 0.4, 0.5],
        'recovery_factor': [0.3, 0.4],
        'fp_thresh': [0.3, 0.35, 0.4],
        'min_track_score': [0.4, 0.5],
        'max_recovery_gap': [2, 3],
    }


def build_config_from_params(params):
    """从参数组合构建tracker配置"""
    from tracker import SequenceTracker
    config = SequenceTracker.default_config()
    
    config['match_thresh']['Box'] = params['match_thresh_box']
    config['match_thresh']['Diver'] = params['match_thresh_diver']
    config['alpha'] = params['alpha']
    config['recovery_score_factor'] = params['recovery_factor']
    config['fp_score_thresh'] = params['fp_thresh']
    config['min_track_score'] = params.get('min_track_score', 0.5)
    config['max_recovery_gap'] = params.get('max_recovery_gap', 3)
    
    return config


def extract_key_metrics(result_dict):
    """提取关键AP指标"""
    metrics = {}
    
    # 3D AP R40 (最重要)
    for cls in ['Car', 'Pedestrian']:
        for diff in ['easy', 'moderate', 'hard']:
            key = f'{cls}_3d/{diff}_R40'
            metrics[key] = result_dict.get(key, 0.0)
    
    # BEV AP R40
    for cls in ['Car', 'Pedestrian']:
        for diff in ['easy', 'moderate', 'hard']:
            key = f'{cls}_bev/{diff}_R40'
            metrics[key] = result_dict.get(key, 0.0)
    
    # 计算聚合指标
    car_3d_avg = sum(metrics.get(f'Car_3d/{d}_R40', 0) for d in ['easy', 'moderate', 'hard']) / 3
    ped_3d_avg = sum(metrics.get(f'Pedestrian_3d/{d}_R40', 0) for d in ['easy', 'moderate', 'hard']) / 3
    
    metrics['Car_3d_avg_R40'] = car_3d_avg
    metrics['Pedestrian_3d_avg_R40'] = ped_3d_avg
    metrics['overall_3d_avg_R40'] = (car_3d_avg + ped_3d_avg) / 2
    
    return metrics


def grid_search(sequence_map, predictions, val_original_frames, data_root,
                search_space=None, class_names=None, verbose=True):
    """
    Grid Search 参数调优
    
    Args:
        sequence_map: 序列映射
        predictions: 原始预测结果
        val_original_frames: val原始帧列表
        data_root: 数据根目录
        search_space: 搜索空间 dict
        class_names: 类名列表
        verbose: 是否打印详细信息
    
    Returns:
        results: list of (params, metrics, config)
        best_result: (params, metrics, config) 最优结果
    """
    from tracker import run_tracking_and_refinement
    from evaluate import (load_gt_infos, filter_original_frames, 
                          predictions_to_eval_annos, run_kitti_eval)
    
    if search_space is None:
        search_space = fast_search_space()
    if class_names is None:
        class_names = ['Box', 'Diver']
    
    # 预加载GT（只加载一次）
    gt_infos_all = load_gt_infos(data_root, 'val')
    gt_infos = filter_original_frames(gt_infos_all, val_original_frames)
    gt_frame_ids = [info['point_cloud']['lidar_idx'] for info in gt_infos]
    
    print(f"GT帧数: {len(gt_frame_ids)}")
    print(f"搜索空间:")
    total_combos = 1
    for key, values in search_space.items():
        print(f"  {key}: {values}")
        total_combos *= len(values)
    print(f"总组合数: {total_combos}\n")
    
    # 生成所有参数组合
    param_keys = list(search_space.keys())
    param_values = list(search_space.values())
    
    results = []
    best_score = -1
    best_result = None
    
    for combo_idx, combo in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_keys, combo))
        
        t0 = time.time()
        config = build_config_from_params(params)
        
        # 运行 tracking + refinement (无需推理，很快)
        refined, _, stats = run_tracking_and_refinement(
            sequence_map, predictions, config
        )
        
        # 评估（仅val原始帧）
        refined_annos = predictions_to_eval_annos(refined, gt_frame_ids, class_names)
        _, result_dict = run_kitti_eval(gt_infos, refined_annos, class_names)
        
        # 提取关键指标
        metrics = extract_key_metrics(result_dict)
        elapsed = time.time() - t0
        
        results.append((params, metrics, config))
        
        score = metrics['overall_3d_avg_R40']
        if score > best_score:
            best_score = score
            best_result = (params, metrics, config)
        
        if verbose:
            print(f"[{combo_idx+1}/{total_combos}] ({elapsed:.1f}s) "
                  f"Car3D={metrics['Car_3d_avg_R40']:.2f} "
                  f"Diver3D={metrics['Pedestrian_3d_avg_R40']:.2f} "
                  f"Overall={score:.2f} "
                  f"{'★ BEST' if score == best_score else ''}")
            if verbose and combo_idx < 5:  # 前几个打印详细参数
                print(f"  params: {params}")
    
    # 打印最优结果
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    best_params, best_metrics, best_config = best_result
    print(f"参数:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print(f"\n指标:")
    for key, val in sorted(best_metrics.items()):
        print(f"  {key}: {val:.4f}")
    
    return results, best_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Search参数调优')
    parser.add_argument('--fast', action='store_true', help='使用精简搜索空间')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    project_root = setup_paths()
    data_root = project_root / 'data' / 'sonar'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'output' / 'temporal_refinement'
    
    # 加载必要数据
    with open(output_dir / 'sequence_map.json', 'r') as f:
        seq_map_data = json.load(f)
    # 兼容新旧格式
    if isinstance(seq_map_data, dict) and 'sequences' in seq_map_data:
        sequence_map = seq_map_data['sequences']
    else:
        sequence_map = seq_map_data  # 旧格式
    
    with open(output_dir / 'raw_predictions.pkl', 'rb') as f:
        predictions = pickle.load(f)
    
    val_original_path = data_root / 'ImageSets' / 'val_original.txt'
    with open(val_original_path, 'r') as f:
        val_original_frames = [line.strip() for line in f.readlines() if line.strip()]
    
    search_space = fast_search_space() if args.fast else default_search_space()
    
    results, best_result = grid_search(
        sequence_map, predictions, val_original_frames, data_root,
        search_space=search_space
    )
    
    # 保存结果
    search_results_path = output_dir / 'grid_search_results.pkl'
    with open(search_results_path, 'wb') as f:
        pickle.dump({
            'results': [(p, m) for p, m, _ in results],
            'best_params': best_result[0],
            'best_metrics': best_result[1],
            'best_config': best_result[2],
        }, f)
    print(f"\n搜索结果保存到: {search_results_path}")
