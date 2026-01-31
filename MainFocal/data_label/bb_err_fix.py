import os
import sys
import numpy as np
import pandas as pd
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

class PointCloudRefinerCentroid:
    """
    点云标注修复工具 (重心法版)
    v1.2 更新：实时进度显示 + np.savetxt 精确控制格式
    """

    def __init__(self, target_classes=[1, 2], iqr_multiplier=3.0, distance_threshold_sigma=3.0):
        """
        Args:
            target_classes: 需要检查的类别
            iqr_multiplier: 筛选异常文件的敏感度
            distance_threshold_sigma: 距离阈值系数 (Mean + K*Std)
        """
        self.target_classes = target_classes
        self.iqr_multiplier = iqr_multiplier
        self.distance_threshold_sigma = distance_threshold_sigma
        self.stats_data = []

    def _compute_pca_max_dim(self, points):
        """计算最大跨度用于筛选文件"""
        if points.shape[0] < 4: return 0.0
        try:
            centroid = np.mean(points, axis=0)
            normalized = points - centroid
            cov = np.cov(normalized, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            projected = np.dot(normalized, vecs)
            extent = np.max(projected, axis=0) - np.min(projected, axis=0)
            return np.max(extent)
        except:
            return 0.0

    def _apply_centroid_filter(self, df, cls_id):
        """
        对特定类别应用重心距离过滤
        """
        mask = df['class_id'] == cls_id
        points = df.loc[mask, ['x', 'y', 'z']].values
        
        if len(points) < 5:
            return False, 0

        # 1. 计算重心
        centroid = np.mean(points, axis=0)
        
        # 2. 计算距离
        dists = np.linalg.norm(points - centroid, axis=1)
        
        # 3. 计算阈值
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        limit = mean_dist + (self.distance_threshold_sigma * std_dist)
        
        # 4. 识别并修改离群点
        is_outlier = dists > limit
        outlier_indices_local = np.where(is_outlier)[0]
        
        if len(outlier_indices_local) == 0:
            return False, 0
            
        indices_to_fix = df.index[mask][outlier_indices_local]
        df.loc[indices_to_fix, 'class_id'] = 3
        
        return True, len(indices_to_fix)

    def run(self, dataset_dir):
        print(f"步骤 1/3: 扫描数据集尺寸分布...")
        
        # --- 阶段 1: 扫描并记录尺寸 (带实时进度) ---
        file_map = {} 
        scan_count = 0
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.txt'):
                    scan_count += 1
                    # 实时打印进度，\r 使其在同一行刷新
                    sys.stdout.write(f"\r  已扫描文件数: {scan_count}")
                    sys.stdout.flush()
                    
                    full_path = os.path.join(root, file)
                    try:
                        # 读取数据
                        df = pd.read_csv(full_path, sep='\s+', header=None, 
                                       names=['x', 'y', 'z', 'intensity', 'class_id'],
                                       dtype={'class_id': float}) # 读取为float以便后续统一格式
                        file_map[full_path] = df
                        
                        for cls_id in self.target_classes:
                            points = df[df['class_id'] == cls_id][['x', 'y', 'z']].values
                            if len(points) > 0:
                                max_dim = self._compute_pca_max_dim(points)
                                self.stats_data.append({
                                    'path': full_path,
                                    'class_id': cls_id,
                                    'max_dim': max_dim
                                })
                    except Exception:
                        pass
        
        print(f"\n  扫描完成。有效数据记录: {len(self.stats_data)}")
        
        if not self.stats_data:
            print("未找到有效数据。")
            return

        # --- 阶段 2: 确定异常文件列表 ---
        df_stats = pd.DataFrame(self.stats_data)
        files_to_fix = set()
        
        print(f"\n步骤 2/3: 识别异常文件 (IQR x {self.iqr_multiplier})...")
        
        for cls_id in self.target_classes:
            subset = df_stats[df_stats['class_id'] == cls_id]
            if len(subset) == 0: continue
            
            Q1 = subset['max_dim'].quantile(0.25)
            Q3 = subset['max_dim'].quantile(0.75)
            IQR = Q3 - Q1
            limit = Q3 + (self.iqr_multiplier * IQR)
            
            outliers = subset[subset['max_dim'] > limit]
            print(f"  Class {cls_id}: 阈值 > {limit:.2f}m (发现 {len(outliers)} 个异常)")
            
            for path in outliers['path'].values:
                files_to_fix.add(path)

        # --- 阶段 3: 执行重心法修复 (使用 np.savetxt) ---
        print(f"\n步骤 3/3: 执行重心距离过滤 (覆盖原文件)...")
        
        fixed_count = 0
        total_points_cleaned = 0
        
        for file_path in files_to_fix:
            try:
                df = file_map[file_path]
                file_modified = False
                temp = 0
                
                for cls_id in self.target_classes:
                    is_changed, count = self._apply_centroid_filter(df, cls_id)
                    if is_changed:
                        file_modified = True
                        total_points_cleaned += count
                        temp = count
                
                if file_modified:
                    # [关键修改] 使用 np.savetxt 保存
                    
                    # 准备数据矩阵 (N, 5)
                    data_to_save = df.values
                    
                    # 定义每一列的格式
                    # 前三列 (x,y,z): '%.6f' (保留6位小数)
                    # 后两列 (intensity, class): '%.1f' (保留1位小数)
                    fmt_list = ['%.6f', '%.6f', '%.6f', '%.1f', '%.1f']
                    
                    # 使用 np.savetxt 覆盖写入
                    # delimiter=' ' 保证以空格分隔
                    np.savetxt(file_path, data_to_save, fmt=fmt_list, delimiter=' ')
                    
                    rel_name = os.path.basename(file_path)
                    print(f"  [已修复] {rel_name}: 清除 {temp} 个点")
                    fixed_count += 1
                    
            except Exception as e:
                print(f"  [Error] {file_path}: {e}")

        print(f"\n任务完成！共修复 {fixed_count} 个文件， 共清除 {total_points_cleaned} 个点")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    DATASET_DIR = r"D:\BaiduNetdiskDownload\datasets" # 请修改为实际路径
    
    refiner = PointCloudRefinerCentroid(
        target_classes=[1, 2], 
        iqr_multiplier=3.0, 
        distance_threshold_sigma=3.0
    )
    
    refiner.run(DATASET_DIR)