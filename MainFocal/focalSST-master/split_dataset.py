import os
import random
import glob
from pathlib import Path

def split_dataset(root_path, split_ratio=0.7):
    # 1. 定义路径
    data_root = Path(root_path)
    points_dir = data_root / 'points'
    imagesets_dir = data_root / 'ImageSets'
    
    # 确保输出目录存在
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 扫描所有 .txt 文件
    # 你的数据没有特定命名规则，所以直接扫描所有 .txt
    print(f"正在扫描目录: {points_dir} ...")
    all_files = glob.glob(str(points_dir / '*.txt'))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"在 {points_dir} 下未找到任何 .txt 文件，请检查路径！")
    
    # 提取文件名（不含扩展名），用作 sample_idx
    sample_ids = [Path(f).stem for f in all_files]
    total_samples = len(sample_ids)
    print(f"共找到 {total_samples} 帧点云数据。")
    
    # 3. 随机打乱
    random.seed(666) # 固定种子，保证复现性
    random.shuffle(sample_ids)
    
    # 4. 划分训练集和验证集
    train_count = int(total_samples * split_ratio)
    train_ids = sample_ids[:train_count]
    val_ids = sample_ids[train_count:]
    
    print(f"训练集数量: {len(train_ids)}")
    print(f"验证集数量: {len(val_ids)}")
    
    # 5. 写入 txt 文件
    def write_list_to_txt(data_list, file_path):
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")
    
    write_list_to_txt(train_ids, imagesets_dir / 'train.txt')
    write_list_to_txt(val_ids, imagesets_dir / 'val.txt')
    
    print(f"数据集划分完成！文件已保存至 {imagesets_dir}")

if __name__ == '__main__':
    # 根据你的描述，数据在 data/sonar
    split_dataset('data/sonar', split_ratio=0.7)