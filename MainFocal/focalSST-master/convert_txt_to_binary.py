"""
将Sonar数据集从文本格式(.txt)转换为二进制格式(.bin)
性能提升: 读取速度5-10倍, 磁盘空间节省75%

使用方法:
    python convert_txt_to_binary.py --data_path data/sonar/points

输出:
    - 创建 data/sonar/points_binary/ 目录
    - 每个.txt文件对应一个.bin文件(float32格式)
    - 生成 conversion_report.txt 记录转换信息
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time


def convert_single_file(txt_file: Path, bin_file: Path):
    """
    将单个.txt文件转换为.bin二进制文件
    
    Args:
        txt_file: 输入的.txt文件路径
        bin_file: 输出的.bin文件路径
    
    Returns:
        (txt_size, bin_size, num_points): 文本大小, 二进制大小, 点数
    """
    # 读取文本数据 [N, 5]: x, y, z, intensity, class_id
    points = np.loadtxt(str(txt_file), dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # 保存为二进制 (float32格式, 4字节/数值)
    points.tofile(str(bin_file))
    
    txt_size = txt_file.stat().st_size
    bin_size = bin_file.stat().st_size
    num_points = points.shape[0]
    
    return txt_size, bin_size, num_points


def main():
    parser = argparse.ArgumentParser(description='将Sonar数据集转换为二进制格式')
    parser.add_argument('--data_path', type=str, 
                        default='data/sonar/points',
                        help='原始.txt文件所在目录')
    parser.add_argument('--output_suffix', type=str,
                        default='_binary',
                        help='输出目录后缀')
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式：只转换前10个文件')
    args = parser.parse_args()
    
    # 设置路径
    txt_dir = Path(args.data_path)
    bin_dir = txt_dir.parent / (txt_dir.name + args.output_suffix)
    
    if not txt_dir.exists():
        raise FileNotFoundError(f"目录不存在: {txt_dir}")
    
    # 创建输出目录
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有.txt文件
    txt_files = sorted(txt_dir.glob('*.txt'))
    
    if args.test_mode:
        txt_files = txt_files[:10]
        print(f"\n⚠️  测试模式：仅转换前10个文件\n")
    
    if len(txt_files) == 0:
        raise FileNotFoundError(f"未找到.txt文件: {txt_dir}")
    
    print(f"\n{'='*70}")
    print(f"📁 输入目录: {txt_dir}")
    print(f"📁 输出目录: {bin_dir}")
    print(f"📄 文件数量: {len(txt_files)}")
    print(f"{'='*70}\n")
    
    # 转换统计
    total_txt_size = 0
    total_bin_size = 0
    total_points = 0
    failed_files = []
    
    start_time = time.time()
    
    # 批量转换
    for txt_file in tqdm(txt_files, desc="转换进度", ncols=80):
        try:
            bin_file = bin_dir / (txt_file.stem + '.bin')
            
            txt_size, bin_size, num_points = convert_single_file(txt_file, bin_file)
            
            total_txt_size += txt_size
            total_bin_size += bin_size
            total_points += num_points
            
        except Exception as e:
            failed_files.append((txt_file.name, str(e)))
            tqdm.write(f"❌ 失败: {txt_file.name} - {e}")
    
    elapsed_time = time.time() - start_time
    
    # 生成报告
    print(f"\n{'='*70}")
    print(f"✅ 转换完成!")
    print(f"{'='*70}")
    print(f"转换文件数: {len(txt_files) - len(failed_files)}/{len(txt_files)}")
    print(f"总点数: {total_points:,}")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print(f"\n--- 存储空间对比 ---")
    print(f"文本格式(.txt):   {total_txt_size / 1024**3:.2f} GB")
    print(f"二进制格式(.bin): {total_bin_size / 1024**3:.2f} GB")
    print(f"节省空间:         {(total_txt_size - total_bin_size) / 1024**3:.2f} GB")
    print(f"压缩率:           {100 * (1 - total_bin_size / total_txt_size):.1f}%")
    print(f"{'='*70}\n")
    
    # 保存详细报告
    report_file = bin_dir.parent / 'conversion_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Sonar数据集格式转换报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"转换时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {txt_dir}\n")
        f.write(f"输出目录: {bin_dir}\n\n")
        f.write(f"--- 统计信息 ---\n")
        f.write(f"文件总数:         {len(txt_files)}\n")
        f.write(f"转换成功:         {len(txt_files) - len(failed_files)}\n")
        f.write(f"转换失败:         {len(failed_files)}\n")
        f.write(f"总点数:           {total_points:,}\n")
        f.write(f"转换耗时:         {elapsed_time:.2f} 秒\n\n")
        f.write(f"--- 存储空间 ---\n")
        f.write(f"文本格式(.txt):   {total_txt_size / 1024**3:.3f} GB\n")
        f.write(f"二进制格式(.bin): {total_bin_size / 1024**3:.3f} GB\n")
        f.write(f"节省空间:         {(total_txt_size - total_bin_size) / 1024**3:.3f} GB\n")
        f.write(f"压缩率:           {100 * (1 - total_bin_size / total_txt_size):.2f}%\n\n")
        
        if failed_files:
            f.write(f"--- 失败文件列表 ---\n")
            for fname, error in failed_files:
                f.write(f"{fname}: {error}\n")
    
    print(f"📄 详细报告已保存: {report_file}\n")
    
    # 速度测试
    print("🚀 进行读取速度测试...\n")
    test_file = txt_files[0]
    test_bin = bin_dir / (test_file.stem + '.bin')
    
    # 测试文本读取
    txt_times = []
    for _ in range(5):
        start = time.time()
        _ = np.loadtxt(str(test_file), dtype=np.float32)
        txt_times.append(time.time() - start)
    txt_avg = np.mean(txt_times)
    
    # 测试二进制读取
    num_points = total_points // len(txt_files)  # 平均点数
    bin_times = []
    for _ in range(5):
        start = time.time()
        _ = np.fromfile(str(test_bin), dtype=np.float32).reshape(-1, 5)
        bin_times.append(time.time() - start)
    bin_avg = np.mean(bin_times)
    
    print(f"文件: {test_file.name}")
    print(f"文本读取(.txt):     {txt_avg*1000:.2f} ms")
    print(f"二进制读取(.bin):   {bin_avg*1000:.2f} ms")
    print(f"速度提升:           {txt_avg/bin_avg:.1f}x\n")
    
    print("="*70)
    print("🎉 全部完成!")
    print(f"请修改配置文件中的 DATA_PATH 为: {bin_dir.parent}")
    print(f"并在 sonar_dataset.yaml 中添加: USE_BINARY_FORMAT: True")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
