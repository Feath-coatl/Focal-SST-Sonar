import numpy as np
import json
import argparse
import os
import platform
import math
from collections import defaultdict
import sys

# ================= 配置区域 =================
# 排除的类别（根据你的 pointcloud_statistics.py 设置）
EXCLUDE_CLASSES = {3, 4}

# 类别名称映射
CLASS_NAMES = {
    1: "木框", 
    2: "蛙人"
}
# ===========================================

class IntensityAnalyzer:
    def __init__(self, root_path):
        self.root_path = self._handle_long_path(root_path)
        self.file_list = []
        
        # 数据存储：使用列表暂存，最后转numpy处理
        # global_all: 包含所有点的intensity (用于查看整体分布，含背景)
        # global_valid: 排除EXCLUDE_CLASSES后的intensity (用于模型训练的实际分布)
        # class_data: 按类别存储 {class_id: [intensities...]}
        self.data_store = {
            "global_all": [],
            "global_valid": [],
            "class_data": defaultdict(list)
        }

    def _handle_long_path(self, path):
        """处理 Windows 长路径问题"""
        abs_path = os.path.abspath(path)
        if platform.system() == "Windows" and not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
        return abs_path

    def scan_files(self):
        """遍历目录下所有txt文件"""
        print(f"正在扫描目录: {self.root_path} ...")
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    self.file_list.append(os.path.join(root, file))
        print(f"共发现 {len(self.file_list)} 个点云文件。")

    def load_data(self):
        """读取所有文件并提取Intensity信息"""
        print("开始读取数据 (这可能需要一些时间)...")
        total_files = len(self.file_list)
        
        for idx, file_path in enumerate(self.file_list):
            if (idx + 1) % 10 == 0:
                print(f"进度: {idx + 1}/{total_files}", end='\r')
            
            try:
                # 假设格式: x y z intensity label
                # 使用 numpy 读取，只取第3列(intensity)和第4列(label)
                # dtype=np.float32 节省内存
                data = np.loadtxt(file_path, usecols=(3, 4), dtype=np.float32)
                
                # 处理单行数据变成一维数组的情况
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                if data.shape[0] == 0:
                    continue

                intensities = data[:, 0]
                labels = data[:, 1]

                # 1. 全局存储
                self.data_store["global_all"].append(intensities)

                # 2. 筛选有效类别（排除背景/噪声）
                # 创建掩码
                valid_mask = np.ones(labels.shape, dtype=bool)
                for excl in EXCLUDE_CLASSES:
                    valid_mask &= (labels != excl)
                
                valid_intensities = intensities[valid_mask]
                if len(valid_intensities) > 0:
                    self.data_store["global_valid"].append(valid_intensities)

                # 3. 按类别存储
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    if label in EXCLUDE_CLASSES:
                        continue
                    # 提取该类别的 intensity
                    cls_mask = (labels == label)
                    cls_intensities = intensities[cls_mask]
                    self.data_store["class_data"][int(label)].append(cls_intensities)

            except Exception as e:
                print(f"\n[Warning] 读取文件出错 {file_path}: {e}")

        print(f"\n数据读取完成。正在合并数据数组...")
        # 将列表合并为 numpy 数组以进行快速计算
        if self.data_store["global_all"]:
            self.data_store["global_all"] = np.concatenate(self.data_store["global_all"])
        else:
            self.data_store["global_all"] = np.array([])

        if self.data_store["global_valid"]:
            self.data_store["global_valid"] = np.concatenate(self.data_store["global_valid"])
        else:
            self.data_store["global_valid"] = np.array([])
            
        for cls_id in self.data_store["class_data"]:
            self.data_store["class_data"][cls_id] = np.concatenate(self.data_store["class_data"][cls_id])

    def _compute_stats(self, data_array, name):
        """计算统计指标的核心函数"""
        if len(data_array) == 0:
            return {"count": 0, "msg": "No data"}

        # 基础统计
        min_val = float(np.min(data_array))
        max_val = float(np.max(data_array))
        mean_val = float(np.mean(data_array))
        std_val = float(np.std(data_array))
        
        # 分位数统计 (关键用于确定截断阈值)
        percentiles = [1, 5, 25, 50, 75, 95, 99, 99.9]
        perc_values = np.percentile(data_array, percentiles)
        perc_dict = {f"p{p}": float(v) for p, v in zip(percentiles, perc_values)}

        # Log 变换统计 (查看 log10(x+1) 后的分布)
        # 防止负数报错，假设 intensity >= 0，如果有负数先clip到0
        safe_data = np.maximum(data_array, 0)
        log_data = np.log10(safe_data + 1.0)
        log_mean = float(np.mean(log_data))
        log_std = float(np.std(log_data))
        
        # 直方图数据 (简略，10个bins)
        hist, bin_edges = np.histogram(data_array, bins=10)
        
        return {
            "name": name,
            "count": int(len(data_array)),
            "basic_stats": {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "median": perc_dict["p50"]
            },
            "percentiles": perc_dict,
            "log_stats_suggestion": {
                "log_mean": log_mean,
                "log_std": log_std
            },
            "distribution_sketch": {
                "bins": [float(x) for x in bin_edges],
                "counts": [int(x) for x in hist]
            }
        }

    def analyze(self):
        """执行全部分析并生成报告"""
        report = {
            "dataset_info": {
                "root_path": self.root_path,
                "total_files_scanned": len(self.file_list)
            },
            "analysis_results": {},
            "normalization_suggestions": []
        }

        print("正在计算统计指标...")

        # 1. Global Valid Stats (最重要，用于决定归一化参数)
        valid_stats = self._compute_stats(self.data_store["global_valid"], "Global (Excluded Noise)")
        report["analysis_results"]["global_valid"] = valid_stats

        # 2. Global All Stats
        all_stats = self._compute_stats(self.data_store["global_all"], "Global (All Points)")
        report["analysis_results"]["global_all"] = all_stats

        # 3. Per Class Stats
        report["analysis_results"]["per_class"] = {}
        for cls_id, data in self.data_store["class_data"].items():
            cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            report["analysis_results"]["per_class"][str(cls_id)] = self._compute_stats(data, cls_name)

        # 4. 生成建议
        suggestions = self._generate_suggestions(valid_stats)
        report["normalization_suggestions"] = suggestions

        return report

    def _generate_suggestions(self, stats):
        """基于统计结果生成中文归一化建议"""
        if stats["count"] == 0:
            return ["没有有效数据，无法给出建议。"]

        sugg = []
        basic = stats["basic_stats"]
        percs = stats["percentiles"]
        
        # 检查数值范围
        max_val = basic["max"]
        p99 = percs["p99"]
        p999 = percs["p99.9"]
        
        sugg.append(f"数据整体范围为 [{basic['min']:.2f}, {basic['max']:.2f}]。")

        # 建议 1: 异常值截断 (Clipping)
        # 如果最大值远大于 p99.9，说明有极值
        if max_val > p999 * 2:
            sugg.append(f"检测到极值 outlier。最大值 ({max_val:.0f}) 远大于 99.9% 分位点 ({p999:.0f})。")
            sugg.append(f"【建议策略 1】: 在归一化前，先将 Intensity 截断(Clip)到 {p999:.1f} 或 {p99:.1f}，以消除离群点影响。")
        else:
            sugg.append(f"极值情况尚可，最大值接近 99.9% 分位点。可以使用 {max_val:.1f} 作为上限。")

        # 建议 2: 归一化方法 (线性 vs Log)
        # 如果数据跨度极大（比如 min=0, max=1e9），且大部分数据集中在低值区（median << mean），建议 Log
        is_highly_skewed = basic["mean"] > basic["median"] * 5  # 简单偏度判断
        is_huge_range = max_val > 10000

        if is_huge_range:
            sugg.append(f"检测到 Intensity 数值量级巨大 (>10000)。")
            if is_highly_skewed:
                sugg.append("数据分布呈现高度右偏（长尾分布）。")
                sugg.append(f"【建议策略 2 (推荐)】: 采用 Log 归一化。公式: input = log10(intensity + 1) / {stats['log_stats_suggestion']['log_mean'] * 2:.2f} (或使用 Log 后的 Max 进行缩放)。")
                sugg.append(f"   - Log变换后，数据均值约为 {stats['log_stats_suggestion']['log_mean']:.2f}。")
            else:
                sugg.append(f"【建议策略 2】: 采用最大值归一化。公式: input = min(intensity, {p999:.1f}) / {p999:.1f}。")
        else:
            sugg.append("数值范围在常规范围内。建议直接使用线性归一化: input = intensity / max_intensity。")

        # 补充：不同类别的差异
        sugg.append("【注意】: 请查看 'per_class' 统计。如果 '蛙人' 和 '木框' 的 Intensity 均值差异巨大，Intensity 将是极佳的分类特征。")

        return sugg

    def save_report(self, report, output_path):
        """保存为易读的 JSON"""
        # 处理 numpy 类型以便 json 序列化
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, cls=NpEncoder, ensure_ascii=False, indent=4)
        print(f"\n分析报告已生成: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description='点云 Intensity 分布分析工具')
    parser.add_argument('--path', type=str, default=r'D:\Desktop\thesis\Modelproject\MainFocal\focalSST-master\data\sonar\points', help='点云文件夹路径')
    parser.add_argument('--output', type=str, default='intensity_analysis_report.json', help='输出报告路径')
    
    args = parser.parse_args()
    
    analyzer = IntensityAnalyzer(args.path)
    analyzer.scan_files()
    
    if not analyzer.file_list:
        print("未找到 .txt 点云文件。")
        return

    analyzer.load_data()
    report = analyzer.analyze()
    analyzer.save_report(report, args.output)

    # 在控制台打印简要建议
    print("\n" + "="*30)
    print("归一化建议摘要:")
    for line in report["normalization_suggestions"]:
        print(f"- {line}")
    print("="*30)

if __name__ == "__main__":
    main()