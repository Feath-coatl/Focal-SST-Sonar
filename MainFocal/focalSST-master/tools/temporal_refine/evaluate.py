"""
Step 5: 序列感知评估
加载GT标注，仅筛选原始val帧，对比Baseline vs Temporal优化的AP。

使用现有 kitti_eval.get_official_eval_result() 评估。
"""

import sys
import copy
import pickle
import numpy as np
from pathlib import Path
from collections import OrderedDict


def setup_paths():
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'tools'))
    return project_root


def load_gt_infos(data_root: Path, split='val'):
    """加载GT info"""
    info_path = data_root / f'sonar_infos_{split}.pkl'
    if not info_path.exists():
        raise FileNotFoundError(f"GT info文件不存在: {info_path}")
    
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    return infos


def filter_original_frames(infos, original_frame_ids):
    """仅保留原始帧"""
    original_set = set(original_frame_ids)
    filtered = [info for info in infos if info['point_cloud']['lidar_idx'] in original_set]
    return filtered


def predictions_to_eval_annos(predictions, frame_ids, class_names):
    """
    将predictions dict转为KITTI eval所需的annos格式
    
    Args:
        predictions: {frame_id: {'boxes_lidar': (N,7), 'score': (N,), 'name': (N,), 'pred_labels': (N,)}}
        frame_ids: 需要评估的帧ID列表（决定了顺序）
        class_names: ['Box', 'Diver']
    
    Returns:
        det_annos: list of dicts，与frame_ids顺序一致
    """
    det_annos = []
    for frame_id in frame_ids:
        pred = predictions.get(frame_id, None)
        
        if pred is None or len(pred['score']) == 0:
            anno = {
                'name': np.array([]),
                'score': np.array([]),
                'boxes_lidar': np.zeros((0, 7)),
                'pred_labels': np.array([]),
                'frame_id': frame_id,
            }
        else:
            anno = {
                'name': pred['name'].copy() if isinstance(pred['name'], np.ndarray) else np.array(pred['name']),
                'score': pred['score'].copy() if isinstance(pred['score'], np.ndarray) else np.array(pred['score']),
                'boxes_lidar': pred['boxes_lidar'].copy() if isinstance(pred['boxes_lidar'], np.ndarray) else np.array(pred['boxes_lidar']),
                'pred_labels': pred['pred_labels'].copy() if isinstance(pred['pred_labels'], np.ndarray) else np.array(pred['pred_labels']),
                'frame_id': frame_id,
            }
        
        det_annos.append(anno)
    
    return det_annos


def run_kitti_eval(gt_infos, det_annos, class_names, map_class_to_kitti=None):
    """
    使用KITTI评估器计算AP
    
    Args:
        gt_infos: GT info列表
        det_annos: 检测annos列表
        class_names: ['Box', 'Diver']
        map_class_to_kitti: 类名映射 {'Box': 'Car', 'Diver': 'Pedestrian'}
    
    Returns:
        result_str: 评估结果字符串
        result_dict: 评估指标字典
    """
    project_root = setup_paths()
    
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    from pcdet.datasets.kitti import kitti_utils
    
    if map_class_to_kitti is None:
        map_class_to_kitti = {
            'Box': 'Car',
            'Diver': 'Pedestrian',
        }
    
    eval_det_annos = copy.deepcopy(det_annos)
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_infos]
    
    # 转换为KITTI格式
    kitti_utils.transform_annotations_to_kitti_format(
        eval_det_annos,
        map_name_to_kitti=map_class_to_kitti
    )
    kitti_utils.transform_annotations_to_kitti_format(
        eval_gt_annos,
        map_name_to_kitti=map_class_to_kitti,
        info_with_fakelidar=False
    )
    
    kitti_class_names = [map_class_to_kitti[x] for x in class_names]
    
    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos=eval_gt_annos,
        dt_annos=eval_det_annos,
        current_classes=kitti_class_names
    )
    
    return ap_result_str, ap_dict


def evaluate_comparison(data_root, baseline_predictions, refined_predictions, 
                        val_original_frames, class_names=None):
    """
    对比评估 Baseline vs Temporal Refinement
    
    Args:
        data_root: data/sonar路径
        baseline_predictions: 原始检测结果 {frame_id: {...}}
        refined_predictions: 优化后检测结果 {frame_id: {...}}
        val_original_frames: val中的原始帧列表
        class_names: 类别列表
    
    Returns:
        comparison: dict 包含两组评估结果
    """
    if class_names is None:
        class_names = ['Box', 'Diver']
    
    # 1. 加载 val GT
    gt_infos_all = load_gt_infos(data_root, 'val')
    
    # 2. 筛选原始帧
    gt_infos = filter_original_frames(gt_infos_all, val_original_frames)
    
    # 确认帧ID对齐
    gt_frame_ids = [info['point_cloud']['lidar_idx'] for info in gt_infos]
    
    print(f"\n评估帧数: {len(gt_frame_ids)} (val原始帧)")
    print(f"GT目标统计:")
    gt_count = {}
    for info in gt_infos:
        for name in info['annos']['name']:
            gt_count[name] = gt_count.get(name, 0) + 1
    for name, count in sorted(gt_count.items()):
        print(f"  {name}: {count}")
    
    # 3. Baseline评估
    print(f"\n{'='*60}")
    print("BASELINE (v4.1 原始检测)")
    print(f"{'='*60}")
    baseline_annos = predictions_to_eval_annos(baseline_predictions, gt_frame_ids, class_names)
    baseline_str, baseline_dict = run_kitti_eval(gt_infos, baseline_annos, class_names)
    print(baseline_str)
    
    # 4. Temporal Refinement评估
    print(f"\n{'='*60}")
    print("TEMPORAL REFINEMENT (后处理优化)")
    print(f"{'='*60}")
    refined_annos = predictions_to_eval_annos(refined_predictions, gt_frame_ids, class_names)
    refined_str, refined_dict = run_kitti_eval(gt_infos, refined_annos, class_names)
    print(refined_str)
    
    # 5. 打印对比表格
    print_comparison_table(baseline_dict, refined_dict)
    
    return {
        'baseline': {'str': baseline_str, 'dict': baseline_dict},
        'refined': {'str': refined_str, 'dict': refined_dict},
    }


def print_comparison_table(baseline_dict, refined_dict):
    """打印便于阅读的对比表格"""
    print(f"\n{'='*80}")
    print("COMPARISON: Baseline vs Temporal Refinement")
    print(f"{'='*80}")
    
    # 提取关键指标
    metrics = [
        ('Car_3d/easy_R40', 'Car 3D AP_R40'),
        ('Pedestrian_3d/easy_R40', 'Diver 3D AP_R40'),
        ('Car_bev/easy_R40', 'Car BEV AP_R40'),
        ('Pedestrian_bev/easy_R40', 'Diver BEV AP@_40')
    ]
    
    print(f"{'Metric':<30} {'Baseline':>10} {'Temporal':>10} {'Delta':>10}")
    print(f"{'-'*60}")
    
    for key, display_name in metrics:
        bl = baseline_dict.get(key, -1)
        rf = refined_dict.get(key, -1)
        if bl >= 0 and rf >= 0:
            delta = rf - bl
            sign = '+' if delta > 0 else ''
            print(f"{display_name:<30} {bl:>10.2f} {rf:>10.2f} {sign}{delta:>9.2f}")
    
    print(f"{'='*80}")


if __name__ == '__main__':
    project_root = setup_paths()
    data_root = project_root / 'data' / 'sonar'
    output_dir = project_root / 'output' / 'temporal_refinement'
    
    # 加载数据
    import json
    
    with open(output_dir / 'sequence_map.json', 'r') as f:
        seq_map_data = json.load(f)
    # 兼容新旧格式
    if isinstance(seq_map_data, dict) and 'sequences' in seq_map_data:
        sequence_map = seq_map_data['sequences']
    else:
        sequence_map = seq_map_data  # 旧格式
    
    with open(output_dir / 'raw_predictions.pkl', 'rb') as f:
        predictions = pickle.load(f)
    
    # 读取val原始帧
    val_original_path = data_root / 'ImageSets' / 'val_original.txt'
    with open(val_original_path, 'r') as f:
        val_original_frames = [line.strip() for line in f.readlines() if line.strip()]
    
    # 运行tracking + refinement
    from tracker import run_tracking_and_refinement
    
    refined_predictions, _, _ = run_tracking_and_refinement(
        sequence_map, predictions
    )
    
    # 评估
    evaluate_comparison(
        data_root, predictions, refined_predictions, val_original_frames
    )
