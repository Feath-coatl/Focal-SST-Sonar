import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
import sys
import platform
import pickle  # 新增: 用于读取pkl

class SuppressStderr:
    """
    上下文管理器：在操作系统层级屏蔽 stderr 输出。
    """
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2)
        os.close(self.null_fds[1])
        os.close(self.save_fds[0])

class PointCloudVisualizer:
    
    def __init__(self, root_path, mode="view", label_path=None):
        # 1. 处理 Windows 长路径问题 (添加 \\?\ 前缀)
        self.root_path = os.path.abspath(root_path)
        if platform.system() == "Windows" and not self.root_path.startswith("\\\\?\\"):
            self.root_path = "\\\\?\\" + self.root_path

        # 新增: 模式与标注路径配置
        self.mode = mode
        self.label_data = {}
        
        # 新增: 加载标注文件
        if self.mode == "label" and label_path:
            try:
                with open(label_path, 'rb') as f:
                    self.label_data = pickle.load(f)
                print(f"成功加载标注文件: {len(self.label_data)} 条记录")
            except Exception as e:
                print(f"无法加载标注文件: {e}")
                self.mode = "view" # 回退到普通模式

        self.file_list = []
        self.current_index = 0
        
        # 数据容器
        self.points = None
        self.intensity = None
        self.labels = None
        self.current_file_path = ""
        self.color_mode = "class" 
        
        # 扫描文件
        self._scan_files()
        
        # 类别配置 (已移除 5,6,7,8 类)
        self.class_names = {
            1: "木框", 
            2: "蛙人", 
            3: "其他噪声", 
            4: "水面"
        }
        
        # 颜色配置 (已移除 5,6,7,8 类)
        self.class_colors = {
            1: [0.55, 0.27, 0.07], # 木框
            2: [0.00, 0.80, 0.00], # 蛙人
            3: [0.50, 0.50, 0.50], # 噪声
            4: [0.00, 0.75, 1.00], # 水面
            0: [0.80, 0.80, 0.80]  # 未定义
        }

    def _scan_files(self):
        """递归扫描，支持 Windows 长路径"""
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
            raise FileNotFoundError(f"在路径下未找到 .txt 文件")
        
        print(f"共找到 {len(self.file_list)} 个数据文件")

    def load_current_data(self):
        self.current_file_path = self.file_list[self.current_index]
        try:
            data = np.loadtxt(self.current_file_path)
            
            if data.size == 0:
                self.points = np.zeros((0, 3))
                self.intensity = np.zeros(0)
                self.labels = np.zeros(0)
                return False

            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            if data.shape[1] < 5:
                return False
            
            self.points = data[:, :3]
            self.intensity = data[:, 3]
            self.labels = data[:, 4].astype(int)
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def _calculate_view_params(self):
        """计算视场角相关参数"""
        if self.points is None or len(self.points) == 0:
            return {
                "min_r": 0.0, "max_r": 0.0,
                "min_theta_deg": 0.0, "max_theta_deg": 0.0,
                "min_phi_deg": 0.0, "max_phi_deg": 0.0
            }
        
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        
        # 计算径向距离 r
        r = np.sqrt(x**2 + y**2 + z**2)
        min_r = np.min(r)
        max_r = np.max(r)
        
        # 计算方位角 theta
        theta_rad = np.arctan2(y, x)
        theta_deg = np.degrees(theta_rad)
        min_theta_deg = np.min(theta_deg)
        max_theta_deg = np.max(theta_deg)
        
        # 计算俯仰角 phi
        r_safe = np.maximum(r, 1e-8) 
        phi_rad = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
        phi_deg = np.degrees(phi_rad)
        min_phi_deg = np.min(phi_deg)
        max_phi_deg = np.max(phi_deg)
        
        return {
            "min_r": min_r, "max_r": max_r,
            "min_theta_deg": min_theta_deg, "max_theta_deg": max_theta_deg,
            "min_phi_deg": min_phi_deg, "max_phi_deg": max_phi_deg
        }

    def _calculate_bbox_size(self):
        """计算轴对齐包围盒（AABB）的尺寸"""
        if self.points is None or len(self.points) == 0:
            return {"x_size": 0.0, "y_size": 0.0, "z_size": 0.0}
        
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)
        
        return {
            "x_min": min_bound[0], "x_max": max_bound[0], 
            "y_min": min_bound[1], "y_max": max_bound[1], 
            "z_min": min_bound[2], "z_max": max_bound[2]
        }

    def get_colors(self):
        if self.points is None or len(self.points) == 0:
            return np.zeros((0, 3))

        if self.color_mode == "class":
            colors = np.zeros((len(self.labels), 3))
            for i, label in enumerate(self.labels):
                colors[i] = self.class_colors.get(label, self.class_colors[0])
            return colors
        else:
            if self.intensity.max() - self.intensity.min() == 0:
                return np.zeros((len(self.intensity), 3))
            
            intensity_norm = (self.intensity - self.intensity.min()) / \
                             (self.intensity.max() - self.intensity.min() + 1e-8)
            
            with SuppressStderr():
                try:
                    colormap = plt.colormaps['jet']
                except AttributeError:
                    colormap = cm.get_cmap('jet')
                return colormap(intensity_norm)[:, :3]

    def create_geometry_list(self):
        if self.points is None or len(self.points) == 0:
            return []

        geometries = []

        # 1. 点云本体
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.get_colors())
        geometries.append(pcd)
        
        # 2. 坐标轴
        point_range = np.max(self.points.max(axis=0) - self.points.min(axis=0))
        frame_size = max(point_range * 0.05, 0.1)
        min_bound = self.points.min(axis=0)
        origin = min_bound - np.array([frame_size, frame_size, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=origin)
        geometries.append(coord_frame)
        
        # 3. 原始的全局 AABB (保留原功能)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        bbox.color = [1, 0, 0] # 红色
        geometries.append(bbox)

        # 4. [新增] 绘制目标 Oriented Bounding Box
        if self.mode == "label" and self.label_data:
            # 获取当前文件的相对路径，用于在字典中查找
            clean_root = self.root_path.replace("\\\\?\\", "")
            clean_current = self.current_file_path.replace("\\\\?\\", "")
            try:
                # 计算相对路径，例如: subfolder/file.txt
                rel_path = os.path.relpath(clean_current, clean_root)
            except ValueError:
                rel_path = clean_current # fallback

            if rel_path in self.label_data:
                targets = self.label_data[rel_path]
                
                # 遍历该文件下的所有类别目标 (例如 class 1 和 2)
                for cls_id, box_info in targets.items():
                    try:
                        # 从字典中提取数据
                        center = box_info['center']
                        extent = box_info['extent']
                        rotation = box_info['rotation']
                        
                        # 创建 Open3D 的 OrientedBoundingBox
                        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
                        
                        # 设置颜色 (使用类别颜色，如果没有则默认黄色)
                        color = self.class_colors.get(cls_id, [1, 1, 0])
                        obb.color = color
                        
                        geometries.append(obb)
                    except KeyError as e:
                        print(f"  [警告] 解析Box数据出错: {e}")

        return geometries

    def print_statistics(self):
        if self.points is None: 
            return

        total_points = len(self.points)
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        view_params = self._calculate_view_params()
        bbox_params = self._calculate_bbox_size()
        
        clean_root = self.root_path.replace("\\\\?\\", "")
        clean_current = self.current_file_path.replace("\\\\?\\", "")
        
        try:
            display_name = os.path.relpath(clean_current, clean_root)
        except ValueError:
            display_name = clean_current

        print("\n" + "="*80)
        print(f"当前文件: {display_name}")
        print(f"进度: [{self.current_index+1}/{len(self.file_list)}]")
        print("="*80)
        print(f"总点数: {total_points}")
        print(f"类别分布:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_points) * 100
            name = self.class_names.get(int(label), f"类别{int(label)}")
            print(f"  [{int(label)}] {name:<5}: {count:>6} 点 ({percentage:>4.1f}%)")
        
        print(f"\n包围盒尺寸:")
        print(f"  X: ({bbox_params['x_min']:>5.2f},{bbox_params['x_max']:>5.2f})")
        print(f"  Y: ({bbox_params['y_min']:>5.2f},{bbox_params['y_max']:>5.2f})")
        print(f"  Z: ({bbox_params['z_min']:>5.2f},{bbox_params['z_max']:>5.2f})")
        
        print(f"\n视场角参数 (角度制):")
        print(f"  径向距离(r): [{view_params['min_r']:>6.2f}, {view_params['max_r']:>6.2f}]")
        print(f"  方位角(θ): [{view_params['min_theta_deg']:>6.2f}°, {view_params['max_theta_deg']:>6.2f}°]")
        print(f"  俯仰角(φ): [{view_params['min_phi_deg']:>6.2f}°, {view_params['max_phi_deg']:>6.2f}°]")
        
        if len(self.intensity) > 0:
            print(f"\n强度范围: [{self.intensity.min():.2f}, {self.intensity.max():.2f}]")

        # 新增: 打印是否有Box被加载
        if self.mode == "label" and display_name in self.label_data:
            print(f"\n[Info] 已加载标注框: 类别 {list(self.label_data[display_name].keys())}")

        print("="*80)
        sys.stdout.flush()

    def switch_file(self, vis, step):
        view_ctl = vis.get_view_control()
        camera_params = view_ctl.convert_to_pinhole_camera_parameters()
        
        self.current_index = (self.current_index + step) % len(self.file_list)
        
        if self.load_current_data():
            self.print_statistics()
            vis.clear_geometries()
            geoms = self.create_geometry_list()
            for geom in geoms:
                vis.add_geometry(geom, reset_bounding_box=False)
            view_ctl.convert_from_pinhole_camera_parameters(camera_params)
        return True

    def toggle_color_mode(self, vis):
        self.color_mode = "intensity" if self.color_mode == "class" else "class"
        view_ctl = vis.get_view_control()
        camera_params = view_ctl.convert_to_pinhole_camera_parameters()
        
        vis.clear_geometries()
        geoms = self.create_geometry_list()
        for geom in geoms:
            vis.add_geometry(geom, reset_bounding_box=False)
        view_ctl.convert_from_pinhole_camera_parameters(camera_params)
        return True

    def run(self):
        print("\n=== 操作说明 ===")
        print("➡️  右方向键: 下一个文件（保持视角）")
        print("⬅️  左方向键: 上一个文件（保持视角）")
        print("C   键: 切换着色模式")
        print("Q   键: 退出")
        if self.mode == "label":
            print("当前模式: 标签预览模式 (显示 3D Box)")
        print("="*60)
        
        if not self.load_current_data():
            print("警告：无法加载第一个文件")

        self.print_statistics()
        
        with SuppressStderr():
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="3D点云可视化工具", width=1400, height=900)

        geoms = self.create_geometry_list()
        for geom in geoms:
            vis.add_geometry(geom)

        vis.register_key_callback(262, lambda v: self.switch_file(v, 1))
        vis.register_key_callback(263, lambda v: self.switch_file(v, -1))
        vis.register_key_callback(ord("C"), lambda v: self.toggle_color_mode(v))

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([1, 1, 1])
        
        with SuppressStderr():
            vis.run()
            
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='3D点云可视化工具（支持3D Box预览）')
    parser.add_argument('--path', type=str, 
                       default=r"D:\Desktop\thesis\Modelproject\MainFocal\focalSST-master\data\sonar\points", 
                       help='点云文件夹路径')
    # 新增参数
    parser.add_argument('--mode', type=str, default='view', choices=['view', 'label'],
                       help='运行模式: view(仅查看), label(查看+显示pkl中的框)')
    parser.add_argument('--label_path', type=str, default=None,
                       help='标签文件(.pkl)的路径，仅在 --mode label 时生效')
    
    args = parser.parse_args()
    
    abs_path = os.path.abspath(args.path)
    if not os.path.exists(abs_path):
        long_path = "\\\\?\\" + abs_path
        if not os.path.exists(long_path):
            print(f"错误: 路径不存在 - {args.path}")
            return
    
    # 检查 label 模式下的参数完整性
    if args.mode == 'label' and not args.label_path:
        print("错误: 使用 --mode label 必须提供 --label_path")
        return
    
    try:
        # 传入新增参数
        viz = PointCloudVisualizer(args.path, mode=args.mode, label_path=args.label_path)
        viz.run()
    except Exception as e:
        print(f"程序运行错误: {e}")

if __name__ == "__main__":
    main()