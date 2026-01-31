import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
import sys
import platform

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

class DualPointCloudVisualizer:
    
    def __init__(self, path_a, path_b):
        # 处理 Windows 长路径
        self.root_path_a = self._fix_path(path_a)
        self.root_path_b = self._fix_path(path_b)

        # 存储匹配好的文件对
        self.matched_files = [] 
        self.current_index = 0
        
        # 定义支持的前缀模式
        self.supported_prefixes = ["a_", "m_", "c_", "d_"]
        self.current_mode_idx = 0 
        self.available_modes = [] 
        
        # 数据容器
        self.data_a = {"points": None, "intensity": None, "labels": None, "path": ""}
        self.data_b = {"points": None, "intensity": None, "labels": None, "path": ""}
        
        self.color_mode = "class" 
        
        # 扫描并匹配文件
        self._scan_and_match_files_smart()
        
        # 类别配置 (已移除 5,6,7,8 类)
        self.class_names = {
            1: "木框", 
            2: "蛙人", 
            3: "背景/噪声", 
            4: "水面"
        }
        
        # 颜色配置 (已移除 5,6,7,8 类)
        self.class_colors = {
            1: [0.55, 0.27, 0.07], # 木框 (棕色)
            2: [0.00, 0.80, 0.00], # 蛙人 (绿色)
            3: [0.50, 0.50, 0.50], # 噪声 (灰色)
            4: [0.00, 0.75, 1.00], # 水面 (天蓝)
            0: [0.80, 0.80, 0.80]  # 未定义
        }

    def _fix_path(self, path):
        abs_path = os.path.abspath(path)
        if platform.system() == "Windows" and not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
        return abs_path

    def _scan_folder(self, root_path):
        """扫描文件夹返回 {相对路径: 绝对路径} 的字典"""
        file_map = {}
        display_path = root_path.replace("\\\\?\\", "")
        print(f"正在扫描: {display_path} ...", end="", flush=True)
        
        if os.path.isfile(root_path):
            file_map[os.path.basename(root_path)] = root_path
        else:
            clean_root_len = len(root_path)
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if file.lower().endswith('.txt'):
                        full_path = os.path.join(root, file)
                        rel_path = full_path[clean_root_len:].lstrip(os.path.sep)
                        file_map[rel_path] = full_path
        print(f" 找到 {len(file_map)} 个文件")
        return file_map

    def _scan_and_match_files_smart(self):
        map_a = self._scan_folder(self.root_path_a)
        print(f"正在分析 Path B 并进行配对...", end="", flush=True)
        
        temp_matches = {} 
        for rel_path, abs_path in map_a.items():
            temp_matches[rel_path] = {
                "path_a": abs_path,
                "variants": {} 
            }
            
        count_variants = 0
        clean_root_b_len = len(self.root_path_b)
        
        for root, dirs, files in os.walk(self.root_path_b):
            for file in files:
                if not file.lower().endswith('.txt'): continue
                
                full_path_b = os.path.join(root, file)
                rel_dir_b = root[clean_root_b_len:].lstrip(os.path.sep)
                
                matched_prefix = None
                original_filename = file
                
                for prefix in self.supported_prefixes:
                    if file.startswith(prefix):
                        matched_prefix = prefix
                        original_filename = file[len(prefix):] 
                        break
                
                candidate_rel_key = os.path.join(rel_dir_b, original_filename)
                
                if candidate_rel_key in temp_matches:
                    key = matched_prefix if matched_prefix else "original_duplicate"
                    temp_matches[candidate_rel_key]["variants"][key] = full_path_b
                    count_variants += 1

        self.matched_files = []
        for rel_name in sorted(temp_matches.keys()):
            item = temp_matches[rel_name]
            item['rel_name'] = rel_name
            self.matched_files.append(item)

        print(f" 完成。")
        print(f"共 {len(self.matched_files)} 个原始文件，找到了 {count_variants} 个增强变体文件。")

    def _load_single_file(self, path, container):
        container["path"] = path if path else "无文件"
        if not path:
            container["points"] = None
            return False
            
        try:
            data = np.loadtxt(path)
            if data.size == 0:
                container["points"] = np.zeros((0, 3))
                container["intensity"] = np.zeros(0)
                container["labels"] = np.zeros(0)
                return False

            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            if data.shape[1] < 5:
                return False
            
            container["points"] = data[:, :3]
            container["intensity"] = data[:, 3]
            container["labels"] = data[:, 4].astype(int)
            return True
        except Exception as e:
            print(f"加载失败 {path}: {e}")
            return False

    def load_current_pair(self):
        if not self.matched_files: return False
        
        current_data = self.matched_files[self.current_index]
        self._load_single_file(current_data["path_a"], self.data_a)
        
        variants = current_data["variants"]
        self.available_modes = sorted(list(variants.keys()))
        
        if not self.available_modes:
            self.data_b["points"] = None
            self.data_b["path"] = "未找到增强文件"
        else:
            if self.current_mode_idx >= len(self.available_modes):
                self.current_mode_idx = 0
            
            mode = self.available_modes[self.current_mode_idx]
            path_b = variants[mode]
            self._load_single_file(path_b, self.data_b)
            
        return True

    def _get_colors(self, container):
        points = container["points"]
        labels = container["labels"]
        intensity = container["intensity"]
        
        if points is None or len(points) == 0:
            return np.zeros((0, 3))

        if self.color_mode == "class":
            colors = np.zeros((len(labels), 3))
            for i, label in enumerate(labels):
                colors[i] = self.class_colors.get(label, self.class_colors[0])
            return colors
        else:
            if len(intensity) == 0 or intensity.max() - intensity.min() == 0:
                return np.zeros((len(intensity), 3))
            
            intensity_norm = (intensity - intensity.min()) / \
                             (intensity.max() - intensity.min() + 1e-8)
            
            with SuppressStderr():
                try:
                    colormap = plt.colormaps['jet']
                except AttributeError:
                    colormap = cm.get_cmap('jet')
                return colormap(intensity_norm)[:, :3]

    def create_geometry_list(self):
        geoms = []
        
        # 记录左侧物体(A)在X轴上的最大值，用于计算间距
        max_x_a = 0.0
        has_a = False
        
        # --- 创建 A (原始 / 左侧) ---
        if self.data_a["points"] is not None and len(self.data_a["points"]) > 0:
            has_a = True
            pcd_a = o3d.geometry.PointCloud()
            pcd_a.points = o3d.utility.Vector3dVector(self.data_a["points"])
            pcd_a.colors = o3d.utility.Vector3dVector(self._get_colors(self.data_a))
            
            # 计算 A 的边界
            min_bound_a = self.data_a["points"].min(axis=0)
            max_bound_a = self.data_a["points"].max(axis=0)
            max_x_a = max_bound_a[0]
            
            bbox_a = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd_a.points)
            bbox_a.color = [1, 0, 0] # 红色
            geoms.append(pcd_a)
            geoms.append(bbox_a)
        
        # --- 创建 B (增强 / 右侧) ---
        # 核心修复: 计算动态偏移量，防止重叠
        # 公式: Translation = (A_Max_X - B_Min_X) + Safety_Gap
        
        if self.data_b["points"] is not None and len(self.data_b["points"]) > 0:
            pcd_b = o3d.geometry.PointCloud()
            points_b = self.data_b["points"] # 原始坐标(未平移)
            
            pcd_b.points = o3d.utility.Vector3dVector(points_b)
            pcd_b.colors = o3d.utility.Vector3dVector(self._get_colors(self.data_b))
            
            # 计算 B 当前(未平移前)的边界
            min_bound_b = points_b.min(axis=0)
            
            # 安全间隙: 固定 5米 或者 A宽度的10%，取大者
            safety_gap = 5.0 
            if has_a:
                width_a = max_bound_a[0] - min_bound_a[0]
                safety_gap = max(5.0, width_a * 0.1)
            
            # 计算所需的平移量
            # 我们希望: (B_Min_X + offset) = A_Max_X + safety_gap
            offset_x = max_x_a - min_bound_b[0] + safety_gap
            
            pcd_b.translate([offset_x, 0, 0])
            
            bbox_b = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd_b.points)
            bbox_b.color = [0, 0, 1] # 蓝色
            geoms.append(pcd_b)
            geoms.append(bbox_b)

        # 坐标轴 (放在 A 的原点)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0])
        geoms.append(coord_frame)
        
        return geoms

    def print_statistics(self):
        clean_name = self.matched_files[self.current_index]["rel_name"]
        
        current_mode_str = "无"
        if self.available_modes:
            current_mode_str = self.available_modes[self.current_mode_idx]

        print("\n" + "="*80)
        print(f"当前原始文件: {clean_name}")
        print(f"文件索引: [{self.current_index+1}/{len(self.matched_files)}] | 视图模式(Mode): {current_mode_str}")
        
        modes_display = []
        for idx, m in enumerate(self.available_modes):
            if idx == self.current_mode_idx:
                modes_display.append(f"[{m}]")
            else:
                modes_display.append(m)
        modes_str = ", ".join(modes_display)
        print(f"可用增强变体: {modes_str if modes_str else '无'}")

        print("-" * 80)
        
        title_b = f"RIGHT (增强: {current_mode_str})"
        print(f"{'LEFT (原始数据)':<38} | {title_b:<38}")
        print("-" * 80)
        
        def get_stats_lines(data_container):
            lines = []
            if data_container["points"] is None:
                lines.append("无数据")
                return lines
            
            fname = os.path.basename(data_container["path"])
            lines.append(f"文件: {fname}")
            
            total = len(data_container["points"])
            lines.append(f"总点数: {total}")
            
            u_labels, counts = np.unique(data_container["labels"], return_counts=True)
            lines.append("类别分布:")
            for l, c in zip(u_labels, counts):
                pct = (c/total)*100
                name = self.class_names.get(int(l), str(int(l)))
                name_str = (name[:6] + '..') if len(name) > 6 else name
                lines.append(f" [{int(l)}] {name_str:<5}: {c:>5} ({pct:>4.1f}%)")
            return lines

        lines_a = get_stats_lines(self.data_a)
        lines_b = get_stats_lines(self.data_b)
        
        max_len = max(len(lines_a), len(lines_b))
        for i in range(max_len):
            str_a = lines_a[i] if i < len(lines_a) else ""
            str_b = lines_b[i] if i < len(lines_b) else ""
            print(f"{str_a:<38} | {str_b:<38}")
            
        print("="*80)
        sys.stdout.flush()

    def switch_file(self, vis, step):
        view_ctl = vis.get_view_control()
        camera_params = view_ctl.convert_to_pinhole_camera_parameters()
        
        self.current_index = (self.current_index + step) % len(self.matched_files)
        
        if self.load_current_pair():
            self.print_statistics()
            vis.clear_geometries()
            geoms = self.create_geometry_list()
            for geom in geoms:
                vis.add_geometry(geom, reset_bounding_box=False)
            view_ctl.convert_from_pinhole_camera_parameters(camera_params)
        return True

    def switch_augment_mode(self, vis):
        if not self.available_modes:
            print("[Info] 当前文件没有增强变体，无法切换模式。")
            return False
            
        view_ctl = vis.get_view_control()
        camera_params = view_ctl.convert_to_pinhole_camera_parameters()
        
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.available_modes)
        
        if self.load_current_pair():
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
        print("\n=== 双视图操作说明 (增强版) ===")
        print("➡️  / ⬅️  : 切换原始文件 (A)")
        print("M        : 切换右侧增强模式 (a_ -> c_ -> d_ -> m_)")
        print("C        : 切换着色 (类别/强度)")
        print("左侧数据 : 原始数据 (红色包围盒)")
        print("右侧数据 : 增强数据 (蓝色包围盒)")
        print("="*60)
        
        if not self.load_current_pair():
            print("警告：无法加载第一组文件")

        self.print_statistics()
        
        with SuppressStderr():
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="Dual PointCloud Viewer", width=1600, height=900)

        geoms = self.create_geometry_list()
        for geom in geoms:
            vis.add_geometry(geom)

        vis.register_key_callback(262, lambda v: self.switch_file(v, 1))
        vis.register_key_callback(263, lambda v: self.switch_file(v, -1))
        vis.register_key_callback(ord("C"), lambda v: self.toggle_color_mode(v))
        vis.register_key_callback(ord("M"), lambda v: self.switch_augment_mode(v)) 

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([1, 1, 1])
        
        with SuppressStderr():
            vis.run()
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='双点云对比可视化工具 (支持增强数据)')
    parser.add_argument('--path_a', type=str, required=True, help='原始数据文件夹 (Left)')
    parser.add_argument('--path_b', type=str, required=True, help='增强数据集文件夹 (Right)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path_a) or not os.path.exists(args.path_b):
        print("错误: 输入的路径不存在")
        return
    
    try:
        viz = DualPointCloudVisualizer(args.path_a, args.path_b)
        viz.run()
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()