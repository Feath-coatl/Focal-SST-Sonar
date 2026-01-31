import os
import numpy as np
import pandas as pd
import warnings

# 忽略 pandas 的一些警告
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    点云标注异常检测器
    通过 PCA 计算目标的真实物理尺寸，并利用统计学方法(IQR)识别尺寸异常过大的文件。
    """

    def __init__(self, target_classes=[1, 2], iqr_multiplier=3.0):
        """
        Args:
            target_classes: 需要检查的类别ID列表
            iqr_multiplier: 离群判定系数。
                            1.5 = 常规离群点 (较严格)
                            3.0 = 极端离群点 (推荐，仅抓取严重错误的)
                            5.0 = 非常宽松，只抓取极其巨大的错误
        """
        self.target_classes = target_classes
        self.iqr_multiplier = iqr_multiplier
        # 用于存储所有样本的尺寸信息
        # 结构: [{'file': path, 'class': id, 'l': len, 'w': wid, 'h': h}, ...]
        self.stats_data = []

    def _compute_pca_dims(self, points):
        """
        使用 PCA 计算点云的主成分尺寸 (Length, Width, Height)
        不关心旋转角度，只关心物体占用的物理体积大小。
        """
        if points.shape[0] < 4:
            return None

        # 1. 归一化
        centroid = np.mean(points, axis=0)
        normalized_points = points - centroid
        
        # 2. PCA 特征分解
        cov_matrix = np.cov(normalized_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 3. 投影到主成分坐标系
        # eigh 返回特征值升序，我们需要降序
        sort_indices = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, sort_indices]
        
        projected_points = np.dot(normalized_points, R)
        
        # 4. 计算真实尺寸 (Extent)
        min_vec = np.min(projected_points, axis=0)
        max_vec = np.max(projected_points, axis=0)
        extent = max_vec - min_vec
        
        # extent = [最大主轴长度, 次轴长度, 最小轴长度]
        return extent

    def scan_dataset(self, dataset_dir):
        """第一阶段：遍历所有文件并计算尺寸"""
        print(f"正在扫描数据集: {dataset_dir} ...")
        file_count = 0
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    self._process_single_file(file_path, dataset_dir)
                    
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"  已分析 {file_count} 个文件...", end='\r')
        
        print(f"\n扫描完成，共获取 {len(self.stats_data)} 个目标样本。")

    def _process_single_file(self, file_path, root_dir):
        try:
            # 仅读取坐标和类别列，加快速度
            df = pd.read_csv(file_path, sep='\s+', header=None, 
                             names=['x', 'y', 'z', 'intensity', 'class_id'],
                             dtype={'class_id': int})
            
            for cls_id in self.target_classes:
                points = df[df['class_id'] == cls_id][['x', 'y', 'z']].values
                if len(points) > 0:
                    extent = self._compute_pca_dims(points)
                    if extent is not None:
                        # 记录数据
                        self.stats_data.append({
                            'filename': os.path.relpath(file_path, root_dir),
                            'full_path': file_path,
                            'class_id': cls_id,
                            'length': extent[0], # 最长边
                            'width': extent[1],  # 中间边
                            'height': extent[2], # 最短边
                            'max_dim': np.max(extent) # 最大跨度(对角线近似)
                        })
        except Exception:
            # 忽略损坏的文件，这里专注于找标注错误
            pass

    def detect_and_report(self):
        """第二阶段：统计分析并输出报告"""
        if not self.stats_data:
            print("未找到任何有效目标数据。")
            return

        df_stats = pd.DataFrame(self.stats_data)
        
        print("\n" + "="*60)
        print("异常检测报告 (基于 PCA 尺寸分析)")
        print("="*60)

        # 针对每个类别分别进行统计
        for cls_id in self.target_classes:
            df_cls = df_stats[df_stats['class_id'] == cls_id]
            
            if len(df_cls) == 0:
                continue
                
            print(f"\n>>> 正在分析类别 Class {cls_id} (样本数: {len(df_cls)})")
            
            # 使用 'max_dim' (最大维度) 作为主要判断依据
            # 因为噪声通常会极大地拉长物体的某一个轴
            data_series = df_cls['max_dim']
            
            # 计算四分位数
            Q1 = data_series.quantile(0.25)
            Q3 = data_series.quantile(0.75)
            IQR = Q3 - Q1
            
            # 设定阈值
            upper_bound = Q3 + (self.iqr_multiplier * IQR)
            
            print(f"    统计特征: Q1={Q1:.2f}m, Q3={Q3:.2f}m, IQR={IQR:.2f}")
            print(f"    自动判定阈值 (Q3 + {self.iqr_multiplier}*IQR): > {upper_bound:.2f} 米")
            
            # 筛选异常文件
            outliers = df_cls[df_cls['max_dim'] > upper_bound].sort_values(by='max_dim', ascending=False)
            
            if len(outliers) > 0:
                print(f"    [!] 发现 {len(outliers)} 个潜在的标注错误文件:")
                print(f"    {'-'*55}")
                print(f"    {'文件名':<35} | {'最大尺寸(米)':<10} | {'超出阈值'}")
                print(f"    {'-'*55}")
                
                for _, row in outliers.iterrows():
                    fname = row['filename']
                    # 截断过长的文件名以便显示
                    if len(fname) > 33: fname = "..." + fname[-30:]
                    
                    dim = row['max_dim']
                    diff = dim - upper_bound
                    print(f"    {fname:<35} | {dim:>8.2f} m  | +{diff:.2f} m")
                
                # 提示
                print(f"\n    建议: 请优先检查上述列表中尺寸最大的前几个文件。")
            else:
                print("    [OK] 该类别未发现明显的尺寸异常文件。")

        print("\n" + "="*60)

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 配置路径
    DATASET_DIR = r"D:\BaiduNetdiskDownload\datasets"  # 请替换为你的数据集路径
    
    # 实例化检测器
    # iqr_multiplier=3.0 表示只检测极端异常值。
    # 如果你想检测更细微的错误，可以改为 1.5
    detector = OutlierDetector(target_classes=[1, 2], iqr_multiplier=3.0)
    
    # 1. 扫描
    detector.scan_dataset(DATASET_DIR)
    
    # 2. 报告
    detector.detect_and_report()