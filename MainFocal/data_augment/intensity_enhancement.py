"""
强度增强脚本
功能：遍历指定文件夹内的.txt数据文件，将目标点(class_id=1,2)的强度增强到
      至少高于75%的背景点(class_id=3,4)强度，且保持目标内部强度的相对关系。
增强后覆盖原文件，仅修改反射强度列。
"""

import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
import augment_utils as utils


def enhance_intensity_for_file(file_path: str, percentile: float = 75.0, 
                               safety_margin: float = 1.1) -> bool:
    """
    对单个文件进行强度增强。
    
    参数:
        file_path: 数据文件路径
        percentile: 背景强度的分位数阈值 (默认75%)
        safety_margin: 安全系数，确保目标强度明显高于阈值 (默认1.1倍)
        
    返回:
        bool: 是否成功增强
    """
    # 读取数据
    data = utils.read_txt(file_path)
    if data is None:
        print(f"[Error] 无法读取文件: {file_path}")
        return False
    
    if data.shape[1] < 5:
        print(f"[Error] 文件格式不正确，列数不足: {file_path}")
        return False
    
    # 提取强度和类别
    intensity = data[:, 3]
    class_id = data[:, 4].astype(int)
    
    # 分离目标点和背景点
    target_mask = np.isin(class_id, [1, 2])  # 木框和蛙人
    background_mask = np.isin(class_id, [3, 4])  # 背景和水面
    
    # 如果没有目标点或背景点，不进行增强
    if not np.any(target_mask):
        return True  # 没有目标点，无需增强
    
    if not np.any(background_mask):
        print(f"[Warning] 文件中没有背景点，跳过增强: {file_path}")
        return True
    
    # 计算背景强度的75%分位数
    background_intensity = intensity[background_mask]
    threshold = np.percentile(background_intensity, percentile)
    
    # 对每类目标分别处理，保持各自内部的相对关系
    enhanced_data = data.copy()
    
    for target_class in [1, 2]:
        class_mask = class_id == target_class
        if not np.any(class_mask):
            continue
        
        # 获取该类目标的强度
        target_intensity = intensity[class_mask]
        
        # 找到该类目标的最小强度
        min_target_intensity = np.min(target_intensity)
        
        # 计算需要的增强因子
        # 目标：使最小强度 * factor > threshold * safety_margin
        target_min = threshold * safety_margin
        
        if min_target_intensity >= target_min:
            # 已经满足要求，无需增强
            continue
        
        # 计算增强因子（加法偏移）
        # 使用加法而不是乘法，以保持相对差异的绝对值
        offset = target_min - min_target_intensity
        
        # 应用增强
        enhanced_data[class_mask, 3] += offset
    
    # 保存增强后的数据，覆盖原文件
    try:
        utils.save_txt(file_path, enhanced_data)
        return True
    except Exception as e:
        print(f"[Error] 保存文件失败 {file_path}: {e}")
        return False


def enhance_directory(directory_path: str, percentile: float = 75.0, 
                      safety_margin: float = 1.1, recursive: bool = True):
    """
    遍历目录下的所有.txt文件并进行强度增强。
    
    参数:
        directory_path: 目标目录路径
        percentile: 背景强度的分位数阈值 (默认75%)
        safety_margin: 安全系数 (默认1.1倍)
        recursive: 是否递归搜索子目录
    """
    # 查找所有.txt文件
    if recursive:
        pattern = os.path.join(directory_path, "**", "*.txt")
        file_list = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory_path, "*.txt")
        file_list = glob.glob(pattern)
    
    if not file_list:
        print(f"[Error] 在目录中未找到.txt文件: {directory_path}")
        return
    
    print(f"[Info] 找到 {len(file_list)} 个文件，开始强度增强...")
    print(f"[Info] 参数: 背景强度分位数={percentile}%, 安全系数={safety_margin}")
    
    success_count = 0
    failed_count = 0
    
    for file_path in tqdm(file_list, desc="强度增强进度"):
        if enhance_intensity_for_file(file_path, percentile, safety_margin):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n[Success] 强度增强完成！")
    print(f"  - 成功: {success_count} 个文件")
    print(f"  - 失败: {failed_count} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="声呐数据强度增强工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python intensity_enhancement.py --path /path/to/dataset
  python intensity_enhancement.py --path /path/to/dataset --percentile 80 --margin 1.2
  python intensity_enhancement.py --path /path/to/dataset --no-recursive
        """
    )
    
    parser.add_argument(
        '--path', 
        type=str, 
        required=True,
        help='包含.txt数据文件的目录路径'
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=75.0,
        help='背景强度的分位数阈值 (默认: 75.0)'
    )
    
    parser.add_argument(
        '--margin',
        type=float,
        default=1.1,
        help='安全系数，确保目标强度明显高于阈值 (默认: 1.1)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录'
    )
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.path):
        print(f"[Error] 路径不存在: {args.path}")
        return
    
    if not os.path.isdir(args.path):
        print(f"[Error] 路径不是目录: {args.path}")
        return
    
    # 执行增强
    enhance_directory(
        args.path,
        percentile=args.percentile,
        safety_margin=args.margin,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()
