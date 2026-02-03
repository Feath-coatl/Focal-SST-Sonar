"""
预处理数据集脚本 - 将体素化结果保存到磁盘
运行一次后，训练时直接加载，连第一个epoch都很快
"""
import os
import sys
import pickle
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def compute_config_hash(data_processor_cfg):
    """计算数据处理配置的哈希值"""
    config_str = str(data_processor_cfg)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None, help='数据根目录')
    args = parser.parse_args()

    # 加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    if args.data_path:
        cfg.DATA_CONFIG.DATA_PATH = args.data_path

    # 临时禁用缓存加载（避免循环依赖）
    cfg.DATA_CONFIG.USE_DISK_CACHE = False
    cfg.DATA_CONFIG.CACHE_ALL_DATA = False
    
    # 计算配置哈希
    config_hash = compute_config_hash(cfg.DATA_CONFIG.DATA_PROCESSOR)
    cache_dir = Path(cfg.DATA_CONFIG.DATA_PATH) / 'preprocessed_cache' / config_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"预处理配置哈希: {config_hash}")
    print(f"缓存目录: {cache_dir}")
    print(f"{'='*60}\n")
    
    # 保存配置信息
    config_info = {
        'config_hash': config_hash,
        'data_processor': cfg.DATA_CONFIG.DATA_PROCESSOR
    }
    with open(cache_dir / 'config.pkl', 'wb') as f:
        pickle.dump(config_info, f)
    
    # 构建数据集
    logger = common_utils.create_logger()
    train_set, _, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        training=True,
        logger=logger
    )
    
    # train_set 可能是 DataLoader 或 Dataset，需要判断
    dataset = train_set.dataset if hasattr(train_set, 'dataset') else train_set
    print(f"数据集大小: {len(dataset)} 样本\n")
    
    # 预处理所有样本
    print("开始预处理...\n")
    for idx in tqdm(range(len(dataset)), desc="预处理进度"):
        try:
            # 获取预处理后的数据
            data_dict = dataset[idx]
            
            # 保存到磁盘
            sample_idx = data_dict['frame_id']
            cache_file = cache_dir / f'{sample_idx}.pkl'
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data_dict, f, protocol=4)  # protocol=4 for better performance
                
        except Exception as e:
            print(f"\n警告: 样本 {idx} 处理失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ 预处理完成！")
    print(f"缓存位置: {cache_dir}")
    print(f"缓存文件数: {len(list(cache_dir.glob('*.pkl'))) - 1}")  # -1 for config.pkl
    print(f"{'='*60}\n")
    
    # 创建完成标记
    (cache_dir / 'COMPLETED').touch()


if __name__ == '__main__':
    main()
