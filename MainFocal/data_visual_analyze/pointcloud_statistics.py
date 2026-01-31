import numpy as np
import json
import re
import argparse
import os
import platform
import sys

# 排除的类别（水面=4，噪声=3）
EXCLUDE_CLASSES = {3, 4}

# 类别名称映射（已剔除 5,6,7,8）
CLASS_NAMES = {
    1: "木框", 
    2: "蛙人"
}

class PointCloudStatistics:
    def __init__(self, root_path):
        # 处理 Windows 长路径
        self.root_path = os.path.abspath(root_path)
        if platform.system() == "Windows" and not self.root_path.startswith("\\\\?\\"):
            self.root_path = "\\\\?\\" + self.root_path
        
        self.file_list = []
        # 核心统计字典（已剔除 5,6,7,8）
        self.stats = {
            "total_files": 0,          # 总文件数
            "total_targets": 0,        # 总目标数（排除3、4类后的实例数）
            "class_statistics": {      # 各类统计
                1: {"name": "木框", "total_points": 0, "instance_count": 0, "average_points": 0.0, "instances": []},
                2: {"name": "蛙人", "total_points": 0, "instance_count": 0, "average_points": 0.0, "instances": []}
            }
        }

    def _scan_files(self):
        """递归扫描txt文件（支持Windows长路径）"""
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
            raise FileNotFoundError(f"在路径 {display_path} 下未找到 .txt 文件")
        
        self.stats["total_files"] = len(self.file_list)
        print(f"共找到 {self.stats['total_files']} 个数据文件")

    def _process_file(self, file_path):
        """处理单个文件，提取统计信息"""
        display_path = file_path.replace("\\\\?\\", "")
        try:
            data = np.loadtxt(file_path)
            
            # 处理空文件
            if data.size == 0:
                print(f"警告: 文件为空 - {display_path}")
                return
            
            # 重塑一维数据
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # 检查列数（至少x,y,z,intensity,label）
            if data.shape[1] < 5:
                print(f"警告: 列数不足 - {display_path} (列数: {data.shape[1]})")
                return
            
            points = data[:, :3]  # x,y,z
            labels = data[:, 4].astype(int)
            
            # 过滤排除的类别
            valid_mask = ~np.isin(labels, list(EXCLUDE_CLASSES))
            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]
            
            # 无有效数据
            if len(valid_labels) == 0:
                return
            
            # 按类别分组统计
            unique_labels = np.unique(valid_labels)
            for label in unique_labels:
                # 跳过不在统计范围内的类别
                if label not in self.stats["class_statistics"]:
                    continue
                
                # 提取该类别的所有点
                class_mask = valid_labels == label
                class_points = valid_points[class_mask]
                point_count = len(class_points)
                
                # 计算中心坐标（x,y,z均值）
                center = class_points.mean(axis=0).tolist()
                center = [round(c, 4) for c in center]  # 保留4位小数
                
                # 更新类统计
                class_stat = self.stats["class_statistics"][label]
                class_stat["total_points"] += point_count
                class_stat["instance_count"] += 1
                self.stats["total_targets"] += 1
                
                # 记录实例信息
                class_stat["instances"].append({
                    "file_path": display_path,
                    "point_count": point_count,
                    "center_xyz": center
                })
                
        except Exception as e:
            print(f"错误: 处理文件失败 - {display_path} (原因: {str(e)})")

    def _calculate_averages(self):
        """计算各类别的平均点数"""
        for label, stat in self.stats["class_statistics"].items():
            if stat["instance_count"] > 0:
                stat["average_points"] = round(
                    stat["total_points"] / stat["instance_count"], 2
                )
            else:
                stat["average_points"] = 0.0

    def _generate_summary(self):
        """生成人类可读的汇总信息"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("点云数据集统计汇总")
        summary.append("="*80)
        summary.append(f"总文件数: {self.stats['total_files']}")
        summary.append(f"总目标数: {self.stats['total_targets']} (排除水面/噪声类别)")
        summary.append("\n各类别统计:")
        summary.append("-"*80)
        
        for label, stat in self.stats["class_statistics"].items():
            summary.append(
                f"[{label}] {stat['name']:8} | 实例数: {stat['instance_count']:4} "
                f"| 总点数: {stat['total_points']:6} | 平均点数: {stat['average_points']:6.2f}"
            )
        
        print("\n".join(summary))
        return "\n".join(summary)

    def run(self, output_path="pointcloud_statistics.json"):
        """主执行流程"""
        # 1. 扫描文件
        self._scan_files()
        
        # 2. 处理所有文件
        print("\n开始处理文件...")
        for i, file_path in enumerate(self.file_list, 1):
            display_path = file_path.replace("\\\\?\\", "")
            print(f"处理进度: [{i}/{self.stats['total_files']}] - {display_path}")
            self._process_file(file_path)
        
        # 3. 计算平均值
        self._calculate_averages()
        
        # 4. 生成汇总
        self._generate_summary()
        
        # 5. 保存 JSON
        json_str = json.dumps(self.stats, ensure_ascii=False, indent=4)
        pattern = r'"center_xyz":\s*\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]'
        replacement = r'"center_xyz": [\1,\2,\3]'
        json_str = re.sub(pattern, replacement, json_str)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        
        print(f"\n统计文件已保存至: {os.path.abspath(output_path)}")
        return self.stats

def main():
    parser = argparse.ArgumentParser(description='点云数据集统计工具')
    parser.add_argument('--path', type=str, 
                       default=r'd:\BaiduNetdiskDownload\datasets', 
                       help='点云文件夹/文件路径')
    parser.add_argument('--output', type=str, 
                       default="pointcloud_statistics.json", 
                       help='统计文件输出路径（JSON格式）')
    
    args = parser.parse_args()
    
    # 验证输入路径
    abs_path = os.path.abspath(args.path)
    long_path = abs_path
    if platform.system() == "Windows" and not abs_path.startswith("\\\\?\\"):
        long_path = "\\\\?\\" + abs_path
    
    if not os.path.exists(long_path):
        print(f"错误: 路径不存在 - {args.path}")
        sys.exit(1)
    
    # 执行统计
    try:
        stats_processor = PointCloudStatistics(args.path)
        stats_processor.run(args.output)
    except Exception as e:
        print(f"程序运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()