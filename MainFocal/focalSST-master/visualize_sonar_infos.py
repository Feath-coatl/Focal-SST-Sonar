import open3d as o3d
import numpy as np
import pickle
import os
import sys
import platform
import traceback
from pathlib import Path
import matplotlib
# 强制 matplotlib 不使用 GUI 后端，防止与 Open3D 冲突
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import cm

# ================= 配置区域 =================
ROOT_PATH = Path('data/sonar')
POINTS_DIR = ROOT_PATH / 'points'

# 类别颜色配置
CLASS_COLORS = {
    'Box': [0.55, 0.27, 0.07], # Brown (Box)
    'Diver': [0.00, 0.80, 0.00], # Green (Diver)
    'Unknown':  [1.00, 0.00, 0.00]  # Red (Unknown)
}
# ===========================================

class AutoVisualizer:
    def __init__(self):
        # 1. 路径检查与处理
        self.root_path = str(POINTS_DIR.resolve())
        # Windows 长路径支持
        if platform.system() == "Windows" and not self.root_path.startswith("\\\\?\\"):
            self.root_path = "\\\\?\\" + self.root_path

        if not os.path.exists(self.root_path):
            print(f"错误: 找不到点云目录: {self.root_path}")
            sys.exit(1)

        # 2. 自动加载 Info 标注文件
        self.gt_database = {} 
        self._load_infos()

        # 3. 初始化文件列表
        self.file_list = []
        self.current_index = 0
        self._scan_files()

        # 4. 数据容器
        self.points = None
        self.intensity = None
        self.labels = None
        self.current_file_path = ""
        self.color_mode = "class" 
        
        self.point_class_colors = {
            1: [0.55, 0.27, 0.07], 2: [0.00, 0.80, 0.00],
            3: [0.50, 0.50, 0.50], 4: [0.00, 0.75, 1.00],
            0: [0.80, 0.80, 0.80]
        }
        self.point_class_names = {1: "木框", 2: "蛙人", 3: "噪声", 4: "水面"}

    def _load_infos(self):
        """
        加载 .pkl 文件
        注意: 数据集已在源头转换为OpenPCDet坐标系,无需运行时坐标变换
        """
        splits = ['train', 'val']
        loaded_count = 0
        print("-" * 50)
        for split in splits:
            pkl_path = ROOT_PATH / f'sonar_infos_{split}.pkl'
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        infos = pickle.load(f)
                    for info in infos:
                        lidar_idx = info['point_cloud']['lidar_idx']
                        annos = info.get('annos', {})
                        if annos is None: continue
                        
                        # 确保数据是 numpy 数组
                        boxes = annos.get('gt_boxes_lidar', np.array([]))
                        names = annos.get('name', np.array([]))
                        
                        if isinstance(boxes, list): boxes = np.array(boxes)
                        if isinstance(names, list): names = np.array(names)
                        
                        # 数据已在源头转换为OpenPCDet坐标系,直接使用
                        if len(boxes) > 0 and boxes.ndim == 1:
                            boxes = boxes.reshape(1, -1)
                            
                        self.gt_database[str(lidar_idx)] = {
                            'boxes': boxes,
                            'names': names
                        }
                    print(f"已加载 {split} 集标注: {len(infos)} 帧")
                    loaded_count += len(infos)
                except Exception as e:
                    print(f"加载 {pkl_path} 失败: {e}")
                    traceback.print_exc()
            else:
                print(f"未找到标注文件: {pkl_path} (跳过)")
        print("-" * 50)

    def _scan_files(self):
        if os.path.isfile(self.root_path):
            self.file_list = [self.root_path]
        else:
            for root, dirs, files in os.walk(self.root_path):
                for file in files:
                    if file.lower().endswith('.txt'):
                        full_path = os.path.join(root, file)
                        self.file_list.append(full_path)
        self.file_list.sort()
        print(f"共找到 {len(self.file_list)} 个点云文件")

    def load_current_data(self):
        self.current_file_path = self.file_list[self.current_index]
        try:
            data = np.loadtxt(self.current_file_path)
            if data.size == 0:
                self.points = np.zeros((0, 3))
                self.intensity = np.zeros(0)
                self.labels = np.zeros(0)
                return False
            if data.ndim == 1: data = data.reshape(1, -1)
            
            # 安全检查列数
            cols = data.shape[1]
            self.points = data[:, :3]
            self.intensity = data[:, 3] if cols > 3 else np.zeros(len(data))
            self.labels = data[:, 4].astype(int) if cols > 4 else np.zeros(len(data))
            return True
        except Exception as e:
            print(f"读取失败 {self.current_file_path}: {e}")
            return False

    def get_colors(self):
        if len(self.points) == 0: return np.zeros((0, 3))
        if self.color_mode == "class":
            colors = np.zeros((len(self.labels), 3))
            for i, label in enumerate(self.labels):
                colors[i] = self.point_class_colors.get(label, self.point_class_colors[0])
            return colors
        else:
            val = self.intensity
            if val.max() - val.min() == 0: return np.zeros((len(val), 3))
            norm = (val - val.min()) / (val.max() - val.min() + 1e-8)
            cmap = plt.get_cmap('jet')
            return cmap(norm)[:, :3]

    def _get_box_geometry(self, box_7, class_name):
        box_7 = box_7.astype(float)
        x, y, z, l, w, h, rot = box_7
        center = [x, y, z]
        extent = [l, w, h]
        c = np.cos(rot)
        s = np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        color = CLASS_COLORS.get(class_name, CLASS_COLORS['Unknown'])
        obb.color = color
        obb.color = np.array(color) # Ensure numpy array
        
        # 绘制线框 (LineSet) 而不是实心盒，便于观察内部点
        # Open3D 的 OBB 默认没有直接转 LineSet 的简单 API，这里创建一个 LineSet
        lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        lines.paint_uniform_color(np.array(color))
        return lines

    def create_geometry_list(self):
        geoms = []
        # 1. 点云
        if self.points is not None and len(self.points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.colors = o3d.utility.Vector3dVector(self.get_colors())
            geoms.append(pcd)
            
            # 坐标系
            min_b = self.points.min(axis=0)
            max_b = self.points.max(axis=0)
            size = max(np.max(max_b - min_b) * 0.05, 0.1)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=min_b - size)
            geoms.append(frame)

        # 2. GT Boxes
        file_name = os.path.basename(self.current_file_path).replace("\\\\?\\", "")
        sample_idx = os.path.splitext(file_name)[0]
        
        if sample_idx in self.gt_database:
            gt_data = self.gt_database[sample_idx]
            boxes = gt_data['boxes']
            names = gt_data['names']
            
            if len(boxes) > 0:
                if boxes.ndim == 1: boxes = boxes.reshape(1, -1)
                for i in range(len(boxes)):
                    try:
                        box = boxes[i]
                        name = names[i] if i < len(names) else 'Unknown'
                        obb_lines = self._get_box_geometry(box, name)
                        geoms.append(obb_lines)
                    except Exception as e:
                        print(f"Box渲染错误 (Index {i}): {e}")
        return geoms

    def print_statistics(self):
        clean_path = self.current_file_path.replace("\\\\?\\", "")
        clean_root = self.root_path.replace("\\\\?\\", "")
        try:
            rel_path = os.path.relpath(clean_path, clean_root)
        except:
            rel_path = clean_path
        file_name = os.path.basename(clean_path)
        sample_idx = os.path.splitext(file_name)[0]
        
        print("\n" + "="*80)
        print(f"文件: {rel_path} | 索引: {sample_idx}")
        print(f"进度: [{self.current_index+1}/{len(self.file_list)}]")
        
        if self.points is not None:
            print(f"点数: {len(self.points)}")
            labels, counts = np.unique(self.labels, return_counts=True)
            for l, c in zip(labels, counts):
                lname = self.point_class_names.get(int(l), str(int(l)))
                print(f"  [{int(l)}] {lname}: {c} ({c/len(self.points)*100:.1f}%)")
                
        if sample_idx in self.gt_database:
            boxes = self.gt_database[sample_idx]['boxes']
            names = self.gt_database[sample_idx]['names'] 
            
            if len(boxes) > 0:
                print(f"\n[GT Boxes] 检测到 {len(boxes)} 个目标 (已还原坐标):")
                if boxes.ndim == 1: boxes = boxes.reshape(1, -1)
                for i in range(len(boxes)):
                    b = boxes[i]
                    n = names[i] if i < len(names) else 'Unknown'
                    print(f"  {i+1}. {n:<8} Loc:({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}) "
                          f"Size:({b[3]:.2f}, {b[4]:.2f}, {b[5]:.2f}) Rot:{b[6]:.2f}rad")
            else:
                print(f"\n[GT Boxes] 当前帧无目标 (背景帧)")
        else:
            print(f"\n[GT Boxes] 未在 .pkl 中找到此帧 ID")
        print("="*80)
        sys.stdout.flush()

    def run(self):
        print("\n=== Sonar Visualizer (Auto Mode) ===")
        print(f"数据集路径: {self.root_path}")
        print("操作: ➡️ 下一个 | ⬅️ 上一个 | C 切换颜色 | Q 退出")
        print("="*60)
        
        if not self.load_current_data():
            print("无法加载第一帧数据")
            
        self.print_statistics()
        
        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="Sonar Data Inspector", width=1400, height=900)
            
            def safe_update(vis, step):
                try:
                    if step != 0:
                        self.current_index = (self.current_index + step) % len(self.file_list)
                        self.load_current_data()
                        self.print_statistics()
                    
                    vis.clear_geometries()
                    for g in self.create_geometry_list():
                        vis.add_geometry(g, reset_bounding_box=False)
                except Exception as e:
                    print(f"更新视图时出错: {e}")
                    traceback.print_exc()
                return True

            for g in self.create_geometry_list():
                vis.add_geometry(g)
                
            vis.register_key_callback(262, lambda v: safe_update(v, 1))
            vis.register_key_callback(263, lambda v: safe_update(v, -1))
            
            def toggle_color(v):
                self.color_mode = "intensity" if self.color_mode == "class" else "class"
                print(f"切换显示模式: {self.color_mode}")
                safe_update(v, 0)
                
            vis.register_key_callback(ord("C"), toggle_color)
            vis.register_key_callback(ord("Q"), lambda v: v.close())
            
            opt = vis.get_render_option()
            opt.point_size = 3.0
            opt.background_color = np.asarray([1, 1, 1])
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print("\n程序发生严重错误:")
            print(error_msg)
            with open("crash_log.txt", "w", encoding='utf-8') as f:
                f.write(error_msg)
            print("错误信息已保存至 crash_log.txt")

if __name__ == "__main__":
    try:
        app = AutoVisualizer()
        app.run()
    except Exception as e:
        with open("crash_log.txt", "w", encoding='utf-8') as f:
            f.write(traceback.format_exc())
        print(f"启动失败: {e}")
        print("详情请查看 crash_log.txt")