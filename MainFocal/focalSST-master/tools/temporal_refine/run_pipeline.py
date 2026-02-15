"""
Step 6: 端到端Pipeline
串联 Step 1→5，一键执行。
支持跳过推理阶段（使用已有的预测结果）。

使用方法:
  cd focalSST-master
  
  # 完整执行（含推理）:
  python tools/temporal_refine/run_pipeline.py
  
  # 跳过推理（已有raw_predictions.pkl）:
  python tools/temporal_refine/run_pipeline.py --skip_inference
  
  # 自定义参数:
  python tools/temporal_refine/run_pipeline.py \
      --cfg_file tools/cfgs/sonar_models/focal_sst_voxelnext_v4.1.yaml \
      --ckpt_path output/sonar_models/focal_sst_voxelnext_v4/voxelnext_v4.1/ckpt/checkpoint_epoch_24.pth \
      --alpha 0.4 --match_thresh_box 3.5 --match_thresh_diver 2.5
"""

import sys
import json
import pickle
import argparse
import time
from pathlib import Path


def setup_paths():
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'tools'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Refinement Pipeline')
    
    # 路径参数
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='模型配置文件（默认: focal_sst_voxelnext_v4.1.yaml）')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoint路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    
    # 流程控制
    parser.add_argument('--skip_inference', action='store_true',
                        help='跳过推理，使用已有的raw_predictions.pkl')
    parser.add_argument('--skip_extract', action='store_true',
                        help='跳过序列提取，使用已有的sequence_map.json')

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    
    # Tracker参数
    parser.add_argument('--match_thresh_box', type=float, default=2.5,
                        help='Box类匹配阈值(米)')
    parser.add_argument('--match_thresh_diver', type=float, default=1.5,
                        help='Diver类匹配阈值(米)')
    parser.add_argument('--max_age', type=int, default=5,
                        help='最大未匹配年龄(帧)')
    
    # Refinement参数
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='置信度校正系数')
    parser.add_argument('--smooth_tau', type=float, default=2.0,
                        help='边框平滑时间常数')
    parser.add_argument('--smooth_window', type=int, default=3,
                        help='边框平滑窗口半径(帧)')
    parser.add_argument('--min_track_len', type=int, default=5,
                        help='最短轨迹长度')
    parser.add_argument('--recovery_factor', type=float, default=0.4,
                        help='恢复帧score缩放因子')
    parser.add_argument('--fp_thresh', type=float, default=0.35,
                        help='误检抑制分数阈值')
    parser.add_argument('--min_track_score', type=float, default=0.5,
                        help='轨迹平均分数阈值(高于此值才恢复)')
    parser.add_argument('--max_recovery_gap', type=int, default=3,
                        help='最大恢复间隔(帧)')
    
    return parser.parse_args()


def build_tracker_config(args):
    """从命令行参数构建tracker配置"""
    from tracker import SequenceTracker
    config = SequenceTracker.default_config()
    
    config['match_thresh']['Box'] = args.match_thresh_box
    config['match_thresh']['Diver'] = args.match_thresh_diver
    config['max_age'] = args.max_age
    config['alpha'] = args.alpha
    config['smooth_tau'] = args.smooth_tau
    config['smooth_window'] = args.smooth_window
    config['min_track_len'] = args.min_track_len
    config['recovery_score_factor'] = args.recovery_factor
    config['fp_score_thresh'] = args.fp_thresh
    config['min_track_score'] = args.min_track_score
    config['max_recovery_gap'] = args.max_recovery_gap
    
    return config


def main():
    start_time = time.time()
    args = parse_args()
    project_root = setup_paths()
    
    # 默认路径
    data_root = project_root / 'data' / 'sonar'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'output' / 'temporal_refinement'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.cfg_file is None:
        args.cfg_file = str(project_root / 'tools' / 'cfgs' / 'sonar_models' / 'focal_sst_voxelnext_v4.1.yaml')
    if args.ckpt_path is None:
        args.ckpt_path = str(project_root / 'output' / 'sonar_models' / 'focal_sst_voxelnext_v4' / 'voxelnext_v4.1' / 'ckpt' / 'checkpoint_epoch_24.pth')
    
    print("=" * 70)
    print("TEMPORAL REFINEMENT PIPELINE")
    print("=" * 70)
    print(f"项目根目录: {project_root}")
    print(f"数据目录:   {data_root}")
    print(f"输出目录:   {output_dir}")
    print(f"配置文件:   {args.cfg_file}")
    print(f"Checkpoint: {args.ckpt_path}")
    print()
    
    # ===== Step 1: 提取原始序列帧 =====
    print("=" * 50)
    print("STEP 1: 提取原始序列帧")
    print("=" * 50)
    
    seq_map_path = output_dir / 'sequence_map.json'
    val_original_path = data_root / 'ImageSets' / 'val_original.txt'
    
    if args.skip_extract and seq_map_path.exists() and val_original_path.exists():
        print("跳过 — 使用已有文件")
        with open(seq_map_path, 'r') as f:
            seq_map_data = json.load(f)
        # 兼容新旧格式
        if isinstance(seq_map_data, dict) and 'sequences' in seq_map_data:
            sequence_map = seq_map_data['sequences']
            print(f"  序列定义: {seq_map_data.get('total_sequences', len(sequence_map))} 个")
        else:
            sequence_map = seq_map_data  # 旧格式
        with open(val_original_path, 'r') as f:
            val_original_frames = [line.strip() for line in f.readlines() if line.strip()]
        print(f"  有效序列: {len(sequence_map)}, val原始帧: {len(val_original_frames)}")
    else:
        from extract_sequences import extract_sequences
        result = extract_sequences(data_root, output_dir)
        sequence_map = result['sequence_map']
        val_original_frames = result['val_original']
    
    step1_time = time.time()
    print(f"\nStep 1 耗时: {step1_time - start_time:.1f}s\n")
    
    # ===== Step 2: 批量推理 =====
    print("=" * 50)
    print("STEP 2: 批量推理")
    print("=" * 50)
    
    pred_path = output_dir / 'raw_predictions.pkl'
    
    if args.skip_inference and pred_path.exists():
        print("跳过推理 — 加载已有预测结果")
        with open(pred_path, 'rb') as f:
            predictions = pickle.load(f)
        print(f"  加载 {len(predictions)} 帧预测结果")
    else:
        from run_inference import run_inference
        predictions = run_inference(
            args.cfg_file, args.ckpt_path, output_dir,
            batch_size=args.batch_size, workers=args.workers
        )
    
    step2_time = time.time()
    print(f"\nStep 2 耗时: {step2_time - step1_time:.1f}s\n")
    
    # ===== Step 3 & 4: Tracking + Refinement =====
    print("=" * 50)
    print("STEP 3 & 4: Tracking + Refinement")
    print("=" * 50)
    
    config = build_tracker_config(args)
    
    # 打印配置
    print("Tracker 参数:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()
    
    from tracker import run_tracking_and_refinement
    
    refined_predictions, all_tracks_by_seq, stats = run_tracking_and_refinement(
        sequence_map, predictions, config
    )
    
    # 保存优化后的预测
    refined_pred_path = output_dir / 'refined_predictions.pkl'
    with open(refined_pred_path, 'wb') as f:
        pickle.dump(refined_predictions, f)
    print(f"\n优化后预测保存到: {refined_pred_path}")
    
    # 保存statistics
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    step34_time = time.time()
    print(f"\nStep 3&4 耗时: {step34_time - step2_time:.1f}s\n")
    
    # ===== Step 5: 评估 =====
    print("=" * 50)
    print("STEP 5: 序列感知评估")
    print("=" * 50)
    
    from evaluate import evaluate_comparison
    
    comparison = evaluate_comparison(
        data_root, predictions, refined_predictions, val_original_frames
    )
    
    # 保存评估结果
    eval_result = {
        'baseline_str': comparison['baseline']['str'],
        'refined_str': comparison['refined']['str'],
        'config': config,
        'stats': stats,
    }
    eval_path = output_dir / 'evaluation_result.pkl'
    with open(eval_path, 'wb') as f:
        pickle.dump(eval_result, f)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Pipeline 完成! 总耗时: {total_time:.1f}s")
    print(f"所有结果保存在: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
