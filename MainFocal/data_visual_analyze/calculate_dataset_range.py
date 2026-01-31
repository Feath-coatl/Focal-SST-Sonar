'''
============================================================
原始数据集全局范围统计结果
============================================================
轴    | 最小值 (Min)     | 最大值 (Max)    | 跨度 (Range)
------------------------------------------------------------
X     | -18.9052        | 19.9508         | 38.8560
Y     | 0.0264          | 47.9698         | 47.9434
Z     | -10.8442        | 13.8041         | 24.6483
------------------------------------------------------------
统计有效文件总数: 3194

============================================================
增强后数据集（含原始）全局范围统计结果
============================================================
轴    | 最小值 (Min)     | 最大值 (Max)    | 跨度 (Range)
------------------------------------------------------------
X     | -19.5764        | 19.9508         | 39.5272
Y     | 0.0264          | 48.6371         | 48.6107
Z     | -19.3662        | 13.8041         | 33.1703
------------------------------------------------------------
统计有效文件总数: 12220
'''


import numpy as np
import argparse
import os
import platform
import sys

class DatasetGlobalRange:
    def __init__(self, root_path, debug_mode=False):
        # 处理 Windows 长路径
        self.root_path = os.path.abspath(root_path)
        if platform.system() == "Windows" and not self.root_path.startswith("\\\\?\\"):
            self.root_path = "\\\\?\\" + self.root_path
        
        self.debug_mode = debug_mode  # 存储调试模式状态
        self.file_list = []
        
        # 初始化全局最小/最大值 [x, y, z]
        self.global_min = np.array([np.inf, np.inf, np.inf])
        self.global_max = np.array([-np.inf, -np.inf, -np.inf])
        self.processed_count = 0
        self.abnormal_files_count = 0  # 记录异常文件数量

    def scan_files(self):
        """递归扫描目录下的所有 .txt 文件"""
        display_path = self.root_path.replace("\\\\?\\", "")
        print(f"正在扫描目录: {display_path}")
        
        if os.path.isfile(self.root_path):
            self.file_list = [self.root_path]
        else:
            for root, dirs, files in os.walk(self.root_path):
                for file in files:
                    if file.lower().endswith('.txt'):
                        full_path = os.path.join(root, file)
                        self.file_list.append(full_path)
        
        self.file_list.sort()
        if not self.file_list:
            print(f"错误: 在路径 {display_path} 下未找到 .txt 文件")
            sys.exit(1)
            
        print(f"共找到 {len(self.file_list)} 个数据文件")

    def _check_y_abnormality(self, file_min, display_path):
        """
        内部调试函数：检查 Y 轴是否小于 0
        file_min: 当前文件的 [min_x, min_y, min_z]
        """
        # 索引 1 对应 Y 轴
        min_y = file_min[1]
        if min_y < 0:
            self.abnormal_files_count += 1
            # 使用醒目的颜色或前缀输出
            print(f" > [DEBUG] 发现异常文件 (Y < 0): {display_path} | Min Y: {min_y:.4f}")

    def process(self):
        """处理所有文件以计算全局范围"""
        print("\n开始计算数据集范围", end="")
        if self.debug_mode:
            print(" (已开启调试模式: 监测 Y < 0)...")
        else:
            print("...")
        
        for i, file_path in enumerate(self.file_list, 1):
            display_path = file_path.replace("\\\\?\\", "")
            
            # 仅在非调试输出或特定进度节点打印进度条，以免淹没调试信息
            if i % 100 == 0 or i == 1 or i == len(self.file_list):
                 print(f"处理进度: [{i}/{len(self.file_list)}] - {display_path}")
            
            try:
                data = np.loadtxt(file_path)
                
                if data.size == 0: continue
                if data.ndim == 1: data = data.reshape(1, -1)
                
                if data.shape[1] < 3:
                    print(f"警告: 文件列数不足 ({data.shape[1]}) - {display_path}")
                    continue
                
                points = data[:, :3]
                
                # 计算当前文件的 min/max
                file_min = np.min(points, axis=0)
                file_max = np.max(points, axis=0)
                
                # --- 调试逻辑 ---
                if self.debug_mode:
                    self._check_y_abnormality(file_min, display_path)
                # ----------------
                
                # 更新全局 min/max
                self.global_min = np.minimum(self.global_min, file_min)
                self.global_max = np.maximum(self.global_max, file_max)
                
                self.processed_count += 1
                
            except Exception as e:
                print(f"错误: 处理文件失败 - {display_path} (原因: {str(e)})")

    def print_results(self):
        """输出统计结果"""
        print("\n" + "="*60)
        print("数据集 XYZ 全局范围统计结果")
        print("="*60)
        
        if self.processed_count == 0:
            print("未处理任何有效文件。")
            return

        axes = ['X', 'Y', 'Z']
        print(f"{'轴':<5} | {'最小值 (Min)':<15} | {'最大值 (Max)':<15} | {'跨度 (Range)':<15}")
        print("-" * 60)
        
        for i, axis in enumerate(axes):
            min_val = self.global_min[i]
            max_val = self.global_max[i]
            span = max_val - min_val
            print(f"{axis:<5} | {min_val:<15.4f} | {max_val:<15.4f} | {span:<15.4f}")
            
        print("-" * 60)
        print(f"统计文件总数: {self.processed_count}")
        
        if self.debug_mode:
             print(f"发现异常文件数 (Y < 0): {self.abnormal_files_count}")
             if self.abnormal_files_count > 0:
                 print("提示: 请检查上述标记为 [DEBUG] 的文件，可能是坐标系定义不同或标定偏移。")
                 
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='点云数据集全局范围统计工具')
    parser.add_argument('--path', type=str, 
                       default=r'D:\Desktop\thesis\Modelproject\MainFocal\focalSST-master\data\sonar\points', 
                       help='点云文件夹路径')
    # 新增调试参数
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式：检测并输出所有 Y < 0 的文件名')
    
    args = parser.parse_args()
    
    check_path = args.path
    if platform.system() == "Windows":
        check_path = os.path.abspath(check_path)
        if not check_path.startswith("\\\\?\\"):
             check_path = "\\\\?\\" + check_path
             
    if not os.path.exists(check_path):
        print(f"错误: 路径不存在 - {args.path}")
        sys.exit(1)

    # 传入 debug 参数
    calculator = DatasetGlobalRange(args.path, debug_mode=args.debug)
    calculator.scan_files()
    calculator.process()
    calculator.print_results()

if __name__ == "__main__":
    main()