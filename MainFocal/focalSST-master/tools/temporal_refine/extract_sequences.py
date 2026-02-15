"""
Step 1: 提取原始序列帧
从 train.txt 和 val.txt 中筛选无前缀的原始帧（排除 a_/d_/m_/c_ 增强帧），
按序列前缀分组并排序，输出序列映射和帧列表文件。

输出:
  - data/sonar/ImageSets/val_original.txt   (val中的原始帧)
  - data/sonar/ImageSets/all_original.txt   (全部原始帧: train+val)
  - output/temporal_refinement/sequence_map.json  (序列映射)
"""

import json
import re
from pathlib import Path
from collections import defaultdict


# 增强帧前缀
AUGMENT_PREFIXES = ('a_', 'd_', 'm_', 'c_')

# 实际序列划分（基于真实数据采集的连续性）
# 每个元组: (序列ID, 起始帧号, 结束帧号)
SEQUENCE_DEFINITIONS = [
    ('seq_01', 130001, 130080),
    ('seq_02', 130081, 130180),
    ('seq_03', 140001, 140394),
    ('seq_04', 140395, 150780),
    ('seq_05', 150781, 150792),
    ('seq_06', 160001, 170075),
    ('seq_07', 170076, 170470),
    ('seq_08', 170471, 170838),
    ('seq_09', 170839, 171042),
    ('seq_10', 171043, 171066),
    ('seq_11', 171067, 171247),
    ('seq_12', 171248, 180158),
]


def is_original_frame(frame_id: str) -> bool:
    """判断是否为原始帧（无增强前缀）"""
    return not any(frame_id.startswith(p) for p in AUGMENT_PREFIXES)


def get_sequence_id(frame_id: str) -> str:
    """
    根据帧号获取所属序列ID
    使用实际序列划分（不是简单的前缀分组）
    """
    try:
        frame_num = int(frame_id)
    except ValueError:
        return None
    
    for seq_id, start, end in SEQUENCE_DEFINITIONS:
        if start <= frame_num <= end:
            return seq_id
    
    return None  # 不在任何已定义序列中


def get_frame_number(frame_id: str) -> int:
    """从帧ID提取帧号（用于排序）"""
    return int(frame_id)


def load_frame_list(txt_path: Path) -> list:
    """加载帧列表文件"""
    if not txt_path.exists():
        raise FileNotFoundError(f"帧列表文件不存在: {txt_path}")
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def extract_sequences(data_root: Path, output_dir: Path):
    """
    提取原始序列帧

    Args:
        data_root: data/sonar 路径
        output_dir: 输出目录 (output/temporal_refinement/)
    """
    imagesets_dir = data_root / 'ImageSets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 train.txt 和 val.txt
    train_frames = load_frame_list(imagesets_dir / 'train.txt')
    val_frames = load_frame_list(imagesets_dir / 'val.txt')

    print(f"总帧数: train={len(train_frames)}, val={len(val_frames)}")

    # 2. 筛选原始帧
    train_original = [f for f in train_frames if is_original_frame(f)]
    val_original = [f for f in val_frames if is_original_frame(f)]
    all_original = sorted(set(train_original + val_original), key=get_frame_number)

    print(f"原始帧数: train={len(train_original)}, val={len(val_original)}, total={len(all_original)}")

    # 3. 按序列分组并排序
    sequence_map = defaultdict(list)
    frames_without_seq = []
    
    for frame_id in all_original:
        seq_id = get_sequence_id(frame_id)
        if seq_id:
            sequence_map[seq_id].append(frame_id)
        else:
            frames_without_seq.append(frame_id)
    
    if frames_without_seq:
        print(f"\n警告: {len(frames_without_seq)} 帧不属于任何定义的序列:")
        for fid in frames_without_seq[:10]:  # 只显示前10个
            print(f"  {fid}")
        if len(frames_without_seq) > 10:
            print(f"  ... 还有 {len(frames_without_seq) - 10} 帧")

    # 排序每个序列内的帧
    for seq_id in sequence_map:
        sequence_map[seq_id] = sorted(sequence_map[seq_id], key=get_frame_number)

    print("\n序列统计:")
    for seq_id, start, end in SEQUENCE_DEFINITIONS:
        frames = sequence_map.get(seq_id, [])
        if frames:
            print(f"  {seq_id} ({start}-{end}): {len(frames)} 帧, 实际范围 {frames[0]}-{frames[-1]}")
        else:
            print(f"  {seq_id} ({start}-{end}): 0 帧 (无数据)")

    # 统计 val 中的原始帧在各序列中的分布
    val_original_set = set(val_original)
    print("\nval原始帧序列分布:")
    for seq_id, start, end in SEQUENCE_DEFINITIONS:
        frames = sequence_map.get(seq_id, [])
        val_in_seq = [f for f in frames if f in val_original_set]
        if frames:
            print(f"  {seq_id}: {len(val_in_seq)} val帧 / {len(frames)} 总帧")

    # 4. 保存输出文件
    # val_original.txt
    val_original_path = imagesets_dir / 'val_original.txt'
    val_original_sorted = sorted(val_original, key=get_frame_number)
    with open(val_original_path, 'w') as f:
        for frame_id in val_original_sorted:
            f.write(f"{frame_id}\n")
    print(f"\n已保存: {val_original_path} ({len(val_original_sorted)} 帧)")

    # all_original.txt
    all_original_path = imagesets_dir / 'all_original.txt'
    with open(all_original_path, 'w') as f:
        for frame_id in all_original:
            f.write(f"{frame_id}\n")
    print(f"已保存: {all_original_path} ({len(all_original)} 帧)")

    # sequence_map.json
    seq_map_path = output_dir / 'sequence_map.json'
    # 添加序列元数据
    seq_map_with_meta = {
        'sequences': dict(sequence_map),
        'definitions': [
            {'id': seq_id, 'start': start, 'end': end}
            for seq_id, start, end in SEQUENCE_DEFINITIONS
        ],
        'total_sequences': len(sequence_map),
        'total_frames': len(all_original),
    }
    with open(seq_map_path, 'w') as f:
        json.dump(seq_map_with_meta, f, indent=2)
    print(f"已保存: {seq_map_path}")

    return {
        'val_original': val_original_sorted,
        'all_original': all_original,
        'sequence_map': dict(sequence_map),
        'frames_without_seq': frames_without_seq,
    }


if __name__ == '__main__':
    # 默认路径（从 tools/ 目录运行）
    project_root = Path(__file__).resolve().parent.parent.parent  # focalSST-master/
    data_root = project_root / 'data' / 'sonar'
    output_dir = project_root / 'output' / 'temporal_refinement'

    result = extract_sequences(data_root, output_dir)
    
    num_valid_seqs = len([s for s in result['sequence_map'].values() if len(s) > 0])
    print(f"\n完成! 共 {len(result['all_original'])} 个原始帧")
    print(f"有效序列: {num_valid_seqs}/{len(SEQUENCE_DEFINITIONS)}")
    if result['frames_without_seq']:
        print(f"未分配帧: {len(result['frames_without_seq'])}")
