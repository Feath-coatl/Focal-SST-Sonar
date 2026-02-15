"""
Step 2: 批量推理所有原始帧
加载v4.1 checkpoint，对所有原始帧逐帧推理，保存predictions。

输入:
  - all_original.txt (来自Step 1)
  - checkpoint_epoch_24.pth

输出:
  - output/temporal_refinement/raw_predictions.pkl
    格式: {frame_id: {'boxes_lidar': (N,7), 'score': (N,), 'name': (N,), 'pred_labels': (N,)}}
"""

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def setup_paths():
    """设置系统路径"""
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'tools'))
    return project_root


def run_inference(cfg_file: str, ckpt_path: str, output_dir: Path, 
                  frame_list_file: str = None, batch_size: int = 1,
                  workers: int = 2):
    """
    对所有原始帧批量推理

    Args:
        cfg_file: 模型配置yaml文件
        ckpt_path: checkpoint路径
        output_dir: 输出目录
        frame_list_file: 帧列表文件路径（若None则使用all_original.txt）
        batch_size: batch大小
        workers: dataloader workers
    """
    project_root = setup_paths()

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    # 1. 加载配置
    cfg_from_yaml_file(cfg_file, cfg)
    
    # 修改配置 → 使用全部原始帧
    if frame_list_file is not None:
        # 将frame_list_file拷贝到ImageSets目录
        frame_list_path = Path(frame_list_file)
        split_name = frame_list_path.stem  # e.g. 'all_original'
    else:
        split_name = 'all_original'
    
    # 修改 DATA_SPLIT 使test指向我们的帧列表
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = split_name
    
    # 修改 INFO_PATH — 我们需要同时加载train和val的info
    # 但实际上推理不需要GT info，只需要点云文件
    # 所以我们自己创建一个临时info list
    cfg.DATA_CONFIG.INFO_PATH['test'] = ['sonar_infos_train.pkl', 'sonar_infos_val.pkl']
    
    # 禁用缓存
    cfg.DATA_CONFIG.CACHE_ALL_DATA = False
    cfg.DATA_CONFIG.USE_DISK_CACHE = False

    # 创建logger
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output_dir / 'inference.log')
    logger.info(f"配置文件: {cfg_file}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # 2. 构建数据集（test模式）
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False
    )

    # 过滤 — 仅保留原始帧
    frame_list_path = Path(cfg.DATA_CONFIG.DATA_PATH) / '..' / '..' / 'data' / 'sonar' / 'ImageSets' / f'{split_name}.txt'
    # 更稳定的路径
    data_root = project_root / 'data' / 'sonar'
    frame_list_path = data_root / 'ImageSets' / f'{split_name}.txt'
    
    if frame_list_path.exists():
        with open(frame_list_path, 'r') as f:
            target_frames = set(line.strip() for line in f.readlines() if line.strip())
        logger.info(f"目标帧数: {len(target_frames)}")
    else:
        target_frames = None
        logger.warning(f"帧列表文件不存在: {frame_list_path}, 将推理所有帧")

    # 过滤数据集中的infos
    if target_frames is not None:
        original_count = len(test_set.sonar_infos)
        test_set.sonar_infos = [
            info for info in test_set.sonar_infos 
            if info['point_cloud']['lidar_idx'] in target_frames
        ]
        logger.info(f"过滤后帧数: {original_count} → {len(test_set.sonar_infos)}")
        
        # 重建 dataloader
        from torch.utils.data import DataLoader
        from functools import partial
        test_loader = DataLoader(
            test_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=False, collate_fn=test_set.collate_batch,
            drop_last=False, timeout=0,
            worker_init_fn=partial(common_utils.worker_init_fn, seed=None)
        )

    # 3. 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    logger.info(f"模型加载完成, 开始推理 {len(test_set)} 帧...")

    # 4. 逐帧推理
    all_predictions = {}
    class_names = cfg.CLASS_NAMES

    with torch.no_grad():
        for batch_dict in tqdm(test_loader, desc="推理中"):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            # 解析每帧结果
            annos = test_set.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names
            )

            for anno in annos:
                frame_id = anno['frame_id']
                all_predictions[frame_id] = {
                    'boxes_lidar': anno['boxes_lidar'].astype(np.float32),
                    'score': anno['score'].astype(np.float32),
                    'name': anno['name'],
                    'pred_labels': anno['pred_labels'].astype(np.int32),
                }

    # 5. 保存结果
    pred_path = output_dir / 'raw_predictions.pkl'
    with open(pred_path, 'wb') as f:
        pickle.dump(all_predictions, f)

    logger.info(f"推理完成! 共 {len(all_predictions)} 帧")
    logger.info(f"结果保存到: {pred_path}")

    # 统计
    total_dets = sum(len(v['score']) for v in all_predictions.values())
    logger.info(f"总检测数: {total_dets}, 平均每帧: {total_dets / max(1, len(all_predictions)):.1f}")

    return all_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量推理原始帧')
    parser.add_argument('--cfg_file', type=str, 
                        default=None,
                        help='模型配置文件')
    parser.add_argument('--ckpt', type=str,
                        default=None,
                        help='Checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    project_root = setup_paths()

    # 默认路径
    if args.cfg_file is None:
        args.cfg_file = str(project_root / 'tools' / 'cfgs' / 'sonar_models' / 'focal_sst_voxelnext_v4.1.yaml')
    if args.ckpt is None:
        args.ckpt = str(project_root / 'output' / 'sonar_models' / 'focal_sst_voxelnext_v4' / 'voxelnext_v4.1' / 'ckpt' / 'checkpoint_epoch_24.pth')

    output_dir = project_root / 'output' / 'temporal_refinement'
    
    run_inference(args.cfg_file, args.ckpt, output_dir, 
                  batch_size=args.batch_size, workers=args.workers)
