import os
import glob
import pickle
import numpy as np
import pandas as pd
import warnings

# 忽略 pandas 的一些 FutureWarning
warnings.filterwarnings('ignore')

class PointCloudBoxGenerator3D:
    """
    点云任意姿态 3D 边界框生成器 (基于 PCA)
    适用于水中目标或非水平放置的目标。
    """

    def __init__(self, target_classes=[1, 2]):
        self.target_classes = target_classes

    def _compute_pca_bbox(self, points):
        """
        使用 PCA (主成分分析) 计算任意姿态的 3D 最小外接矩形。
        
        Args:
            points: Nx3 numpy array (x, y, z)
            
        Returns:
            dict: {
                'center': np.array([cx, cy, cz]),
                'extent': np.array([length, width, height]), # 对应主轴、次轴、第三轴的长度
                'rotation': 3x3 np.array (旋转矩阵)
            }
        """
        if points.shape[0] < 4:
            # 点数太少无法构建有效的3D体
            return None

        # 1. 计算质心
        centroid = np.mean(points, axis=0)
        
        # 2. 去中心化
        normalized_points = points - centroid
        
        # 3. 计算协方差矩阵并进行特征分解
        # rowvar=False 表示每一列代表一个变量(x, y, z)
        cov_matrix = np.cov(normalized_points, rowvar=False)
        
        # eigenvalues: 特征值 (代表方差大小)
        # eigenvectors: 特征向量 (代表主方向，列向量)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. 排序特征向量
        # eigh 返回的特征值是升序排列的，我们需要降序 (最大方差为主轴)
        # argsort 返回索引，[::-1] 反转索引
        sort_indices = np.argsort(eigenvalues)[::-1]
        
        # 重新排列特征向量，构成旋转矩阵 R (基向量)
        # R 的每一列都是一个新的坐标轴方向
        R = eigenvectors[:, sort_indices]
        
        # 确保坐标系是右手系 (通过叉乘检查)
        # 如果 z 轴方向与 x cross y 相反，则反转 z 轴
        if np.dot(np.cross(R[:, 0], R[:, 1]), R[:, 2]) < 0:
            R[:, 2] = -R[:, 2]

        # 5. 将原始点云投影到新的主成分坐标系 (Eigenbasis)
        # transform: P_new = (P_old - centroid) @ R
        projected_points = np.dot(normalized_points, R)
        
        # 6. 在新坐标系下计算 AABB (Axis Aligned Bounding Box)
        min_vec = np.min(projected_points, axis=0)
        max_vec = np.max(projected_points, axis=0)
        
        # 7. 计算尺寸 (extent)
        extent = max_vec - min_vec
        
        # 8. 计算新坐标系下的中心，并转换回世界坐标系
        # 新坐标系下的中心位置
        center_local = (max_vec + min_vec) / 2.0
        
        # 转换回世界坐标: center_world = centroid + R @ center_local
        center_world = centroid + np.dot(R, center_local)

        return {
            'center': center_world,   # [x, y, z]
            'extent': extent,         # [l, w, h] (沿主轴、次轴、第三轴的尺寸)
            'rotation': R             # 3x3 旋转矩阵，列向量为局部坐标轴的方向
        }

    def process_file(self, file_path):
        """
        处理单个文件
        """
        results = {}
        try:
            # 读取数据
            df = pd.read_csv(file_path, sep='\s+', header=None, 
                             names=['x', 'y', 'z', 'intensity', 'class_id'],
                             dtype={'class_id': int})
            
            for cls_id in self.target_classes:
                # 提取特定类别的点
                target_points = df[df['class_id'] == cls_id][['x', 'y', 'z']].values
                
                if len(target_points) > 0:
                    bbox = self._compute_pca_bbox(target_points)
                    if bbox:
                        results[cls_id] = bbox
                        
        except Exception as e:
            print(f"[Error] 处理文件失败 {file_path}: {e}")
            return None
            
        return results if results else None

    def run(self, dataset_dir, output_pkl_path):
        """
        主运行函数
        """
        all_labels = {}
        file_count = 0
        
        print(f"开始处理数据集 (3D任意姿态): {dataset_dir}")
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dataset_dir)
                    
                    bbox_res = self.process_file(file_path)
                    
                    if bbox_res:
                        all_labels[rel_path] = bbox_res
                    
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"已处理 {file_count} 个文件...")

        # 保存结果
        try:
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(all_labels, f)
            print(f"\n处理完成！共扫描 {file_count} 个文件。")
            print(f"有效标注已保存至: {output_pkl_path}")
            
            # 打印一个示例看看数据结构
            if len(all_labels) > 0:
                first_key = list(all_labels.keys())[0]
                print(f"\n示例数据 ({first_key}):")
                print(all_labels[first_key])
                
        except Exception as e:
            print(f"保存 PKL 文件失败: {e}")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置路径
    DATASET_DIR = r"D:\Desktop\thesis\Modelproject\MainFocal\focalSST-master\data\sonar\points"
    OUTPUT_PATH = r"D:\Desktop\thesis\Modelproject\MainFocal\focalSST-master\labels_3d.pkl"      
    
    generator = PointCloudBoxGenerator3D()
    generator.run(DATASET_DIR, OUTPUT_PATH)
    #print("请配置 DATASET_DIR 后运行代码。")