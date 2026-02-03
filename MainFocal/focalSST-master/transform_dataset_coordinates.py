"""
坐标系转换脚本 - 从源头修改数据集
将Sonar原始坐标系 (X右, Y前, Z上) 转换为 OpenPCDet坐标系 (X前, Y左, Z上)

坐标变换:
  - new_X = old_Y  (前向)
  - new_Y = -old_X (左向)
  - new_Z = old_Z  (上向)
  - intensity保持不变
  - class保持不变

作者: 用于一次性转换所有数据集坐标系
使用前请备份原始数据!
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

def transform_coordinate_system(old_points):
    """
    转换坐标系统
    
    Args:
        old_points: [N, 5] - [x, y, z, intensity, class]
    
    Returns:
        new_points: [N, 5] - [new_x, new_y, new_z, intensity, class]
    """
    new_points = np.zeros_like(old_points)
    
    # 坐标变换: Sonar系统 -> OpenPCDet系统
    new_points[:, 0] = old_points[:, 1]  # new_X = old_Y (前向)
    new_points[:, 1] = -old_points[:, 0]  # new_Y = -old_X (左向)
    new_points[:, 2] = old_points[:, 2]   # new_Z = old_Z (上向)
    new_points[:, 3] = old_points[:, 3]   # intensity保持不变
    new_points[:, 4] = old_points[:, 4]   # class保持不变
    
    return new_points

def transform_single_file(file_path, backup_dir=None):
    """
    转换单个TXT文件的坐标系统
    
    Args:
        file_path: Path对象,指向TXT文件
        backup_dir: 可选,备份目录路径
    
    Returns:
        success: bool, 是否成功
    """
    try:
        # 备份原始文件(如果指定了备份目录)
        if backup_dir is not None:
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
        
        # 读取原始数据
        old_points = np.loadtxt(str(file_path), dtype=np.float32)
        
        # 处理单点情况
        if old_points.ndim == 1:
            old_points = old_points.reshape(1, -1)
        
        # 检查数据维度
        if old_points.shape[1] != 5:
            print(f"警告: {file_path.name} 的列数为 {old_points.shape[1]}, 期望为5")
            return False
        
        # 转换坐标
        new_points = transform_coordinate_system(old_points)
        
        # 原地覆盖写入
        np.savetxt(str(file_path), new_points, fmt='%.6f %.6f %.6f %.1f %.1f')
        
        return True
        
    except Exception as e:
        print(f"处理 {file_path.name} 时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='转换Sonar数据集坐标系统')
    parser.add_argument('--data_path', type=str, 
                        default='data/sonar/points',
                        help='点云数据目录路径')
    parser.add_argument('--backup', action='store_true',
                        help='是否备份原始数据到backup文件夹')
    parser.add_argument('--dry_run', action='store_true',
                        help='试运行模式,不实际修改文件')
    
    args = parser.parse_args()
    
    # 获取点云数据目录
    points_dir = Path(args.data_path)
    if not points_dir.exists():
        print(f"错误: 目录不存在 - {points_dir}")
        return
    
    # 创建备份目录(如果需要)
    backup_dir = None
    if args.backup:
        backup_dir = points_dir.parent / 'points_backup_original'
        backup_dir.mkdir(exist_ok=True)
        print(f"备份目录: {backup_dir}")
    
    # 获取所有TXT文件
    txt_files = sorted(list(points_dir.glob('*.txt')))
    print(f"找到 {len(txt_files)} 个点云文件")
    
    if len(txt_files) == 0:
        print("警告: 没有找到任何TXT文件")
        return
    
    # 试运行模式
    if args.dry_run:
        print("\n=== 试运行模式 ===")
        # 只处理前3个文件作为示例
        for i, file_path in enumerate(txt_files[:3]):
            print(f"\n处理示例 {i+1}: {file_path.name}")
            old_points = np.loadtxt(str(file_path), dtype=np.float32)
            if old_points.ndim == 1:
                old_points = old_points.reshape(1, -1)
            
            new_points = transform_coordinate_system(old_points)
            
            print(f"原始坐标 (前5行):\n{old_points[:5, :3]}")
            print(f"转换后坐标 (前5行):\n{new_points[:5, :3]}")
            print(f"点云数量: {len(old_points)}")
        
        print("\n试运行完成。使用不带 --dry_run 参数来实际执行转换。")
        return
    
    # 确认操作
    print("\n警告: 此操作将原地修改所有点云文件的坐标系统!")
    if args.backup:
        print(f"原始文件将备份到: {backup_dir}")
    else:
        print("未启用备份,原始数据将被覆盖!")
    
    response = input("\n确认继续? (yes/no): ")
    if response.lower() != 'yes':
        print("操作已取消")
        return
    
    # 批量处理文件
    print("\n开始转换...")
    success_count = 0
    fail_count = 0
    
    for file_path in tqdm(txt_files, desc="转换进度"):
        if transform_single_file(file_path, backup_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出统计信息
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    
    # 验证转换结果
    if success_count > 0:
        print("\n验证前3个文件的转换结果...")
        for file_path in txt_files[:3]:
            points = np.loadtxt(str(file_path), dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            print(f"\n{file_path.name}:")
            print(f"  点数: {len(points)}")
            print(f"  X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print("\n=== 后续步骤 ===")
    print("1. 运行: python create_sonar_infos.py 重新生成标注文件")
    print("2. 检查 visualize_sonar_infos.py 和 sonar_dataset.py 中的坐标转换代码")
    print("3. 删除或注释掉运行时的坐标变换逻辑")

if __name__ == '__main__':
    main()
