#!/usr/bin/env python3
"""
三视图检测对比可视化工具
同时显示: GT BOX | 原始检测 | 后处理优化
"""

import open3d as o3d
import numpy as np
import pickle
import os
import sys
import platform
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 配置区域 =================
ROOT_PATH = Path('../../data/sonar')
POINTS_DIR = ROOT_PATH / 'points'
OUTPUT_DIR = Path('../../output/temporal_refinement')

# 视图间距配置
HORIZONTAL_GAP = 10.0  # 三个视图之间的水平间距(米)

# 检测框颜色配置
BOX_COLORS = {
    'gt': {
        'Box': [0.0, 0.8, 0.0],    # 绿色 - GT Box
        'Diver': [0.0, 0.4, 0.0],  # 青色 - GT Diver
    },
    'raw': {
        'Box': [0.9, 0.0, 0.0],    # 橙色 - 原始检测 Box
        'Diver': [0.4, 0.0, 0.0],  # 品红 - 原始检测 Diver
    },
    'temporal': {
        'Box': [0.0, 0.0, 0.9],    # 红色 - 后处理 Box
        'Diver': [0.0, 0.0, 0.4],  # 紫色 - 后处理 Diver
    }
}

# 点云着色方案
POINT_CLASS_COLORS = {
    1: [0.55, 0.27, 0.07],  # 木框
    2: [0.00, 0.80, 0.00],  # 蛙人
    3: [0.50, 0.50, 0.50],  # 噪声
    4: [0.00, 0.75, 1.00],  # 水面
    0: [0.80, 0.80, 0.80]   # 未定义
}

POINT_CLASS_NAMES = {1: "木框", 2: "蛙人", 3: "噪声", 4: "水面"}
# ===========================================


class TripleViewVisualizer:
    def __init__(self):
        # 路径处理
        self.root_path = str(POINTS_DIR.resolve())
        if platform.system() == "Windows" and not self.root_path.startswith("\\\\?\\"):
            self.root_path = "\\\\?\\" + self.root_path

        if not os.path.exists(self.root_path):
            print(f"错误: 找不到点云目录: {self.root_path}")
            sys.exit(1)

        # 加载数据
        self.gt_database = {}
        self.raw_predictions = {}
        self.refined_predictions = {}
        
        self._load_gt_annotations()
        self._load_predictions()

        # 扫描val原始帧
        self.val_frames = self._load_val_frames()
        self.current_index = 0

        # 数据容器
        self.points = None
        self.intensity = None
        self.labels = None
        self.current_frame_id = ""
        self.color_mode = "class"
        
        print(f"\n初始化完成:")
        print(f"  GT标注: {len(self.gt_database)} 帧")
        print(f"  原始检测: {len(self.raw_predictions)} 帧")
        print(f"  后处理检测: {len(self.refined_predictions)} 帧")
        print(f"  Val帧: {len(self.val_frames)} 帧")

    def _load_gt_annotations(self):
        """加载GT标注"""
        splits = ['train', 'val']
        for split in splits:
            pkl_path = ROOT_PATH / f'sonar_infos_{split}.pkl'
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        infos = pickle.load(f)
                    for info in infos:
                        lidar_idx = info['point_cloud']['lidar_idx']
                        annos = info.get('annos', {})
                        if annos is None:
                            continue
                        
                        boxes = annos.get('gt_boxes_lidar', np.array([]))
                        names = annos.get('name', np.array([]))
                        
                        if isinstance(boxes, list):
                            boxes = np.array(boxes)
                        if isinstance(names, list):
                            names = np.array(names)
                        
                        if len(boxes) > 0 and boxes.ndim == 1:
                            boxes = boxes.reshape(1, -1)
                            
                        self.gt_database[str(lidar_idx)] = {
                            'boxes': boxes,
                            'names': names
                        }
                    print(f"✓ 加载GT {split}: {len(infos)} 帧")
                except Exception as e:
                    print(f"✗ 加载GT {pkl_path} 失败: {e}")

    def _load_predictions(self):
        """加载检测结果"""
        # 原始检测
        raw_path = OUTPUT_DIR / 'raw_predictions.pkl'
        if raw_path.exists():
            try:
                with open(raw_path, 'rb') as f:
                    raw_data = pickle.load(f)
                
                # 处理不同格式
                if isinstance(raw_data, dict):
                    # 字典格式: {frame_id: {boxes, names, scores, ...}}
                    self.raw_predictions = raw_data
                    print(f"✓ 加载原始检测: {len(raw_data)} 帧 (字典格式)")
                elif isinstance(raw_data, list):
                    # 列表格式: [{frame_id, boxes, names, ...}, ...]
                    for item in raw_data:
                        frame_id = item['frame_id']
                        self.raw_predictions[frame_id] = item
                    print(f"✓ 加载原始检测: {len(raw_data)} 帧 (列表格式)")
            except Exception as e:
                print(f"✗ 加载原始检测失败: {e}")
        else:
            print(f"⚠ 未找到原始检测文件: {raw_path}")

        # 后处理检测
        temporal_path = OUTPUT_DIR / 'refined_predictions.pkl'
        if temporal_path.exists():
            try:
                with open(temporal_path, 'rb') as f:
                    temporal_data = pickle.load(f)
                
                # 处理不同格式
                if isinstance(temporal_data, dict):
                    self.refined_predictions = temporal_data
                    print(f"✓ 加载后处理检测: {len(temporal_data)} 帧 (字典格式)")
                elif isinstance(temporal_data, list):
                    for item in temporal_data:
                        frame_id = item['frame_id']
                        self.refined_predictions[frame_id] = item
                    print(f"✓ 加载后处理检测: {len(temporal_data)} 帧 (列表格式)")
            except Exception as e:
                print(f"✗ 加载后处理检测失败: {e}")
        else:
            print(f"⚠ 未找到后处理文件: {temporal_path}")

    def _load_val_frames(self):
        """加载val原始帧列表"""
        val_original_path = ROOT_PATH / 'ImageSets' / 'val_original.txt'
        if val_original_path.exists():
            with open(val_original_path, 'r') as f:
                frames = [line.strip() for line in f if line.strip()]
            return sorted(frames)
        else:
            print(f"⚠ 未找到val_original.txt，使用所有GT帧")
            return sorted(list(self.gt_database.keys()))

    def load_current_frame(self):
        """加载当前帧点云"""
        self.current_frame_id = self.val_frames[self.current_index]
        point_file = os.path.join(self.root_path, f"{self.current_frame_id}.txt")
        
        try:
            data = np.loadtxt(point_file)
            if data.size == 0:
                self.points = np.zeros((0, 3))
                self.intensity = np.zeros(0)
                self.labels = np.zeros(0)
                return False
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            cols = data.shape[1]
            self.points = data[:, :3]
            self.intensity = data[:, 3] if cols > 3 else np.zeros(len(data))
            self.labels = data[:, 4].astype(int) if cols > 4 else np.zeros(len(data))
            return True
        except Exception as e:
            print(f"读取失败 {point_file}: {e}")
            return False

    def _get_point_colors(self):
        """获取点云颜色"""
        if len(self.points) == 0:
            return np.zeros((0, 3))
        
        if self.color_mode == "class":
            colors = np.zeros((len(self.labels), 3))
            for i, label in enumerate(self.labels):
                colors[i] = POINT_CLASS_COLORS.get(label, POINT_CLASS_COLORS[0])
            return colors
        else:
            val = self.intensity
            if val.max() - val.min() == 0:
                return np.zeros((len(val), 3))
            norm = (val - val.min()) / (val.max() - val.min() + 1e-8)
            cmap = plt.get_cmap('jet')
            return cmap(norm)[:, :3]

    def _create_box_lineset(self, box_7, color):
        """创建检测框线集"""
        try:
            x, y, z, l, w, h, rot = box_7.astype(float)
            center = [x, y, z]
            extent = [l, w, h]
            c = np.cos(rot)
            s = np.sin(rot)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

            # 确保颜色是numpy array且在[0,1]范围内
            color_array = np.array(color, dtype=np.float64)
            color_array = np.clip(color_array, 0.0, 1.0)  # 裁剪到[0,1]
            lines.paint_uniform_color(color_array)
            return lines
        except Exception as e:
            print(f"创建box失败: {e}")
            return None

    def _create_single_view(self, offset_x, view_type):
        """
        创建单个视图
        view_type: 'gt', 'raw', 'temporal'
        """
        geoms = []
        
        # 1. 点云
        if self.points is not None and len(self.points) > 0:
            pcd = o3d.geometry.PointCloud()
            points_shifted = self.points.copy()
            points_shifted[:, 0] += offset_x
            pcd.points = o3d.utility.Vector3dVector(points_shifted)
            pcd.colors = o3d.utility.Vector3dVector(self._get_point_colors())
            geoms.append(pcd)

        # 2. 根据类型添加boxes
        boxes = []
        names = []
        
        if view_type == 'gt' and self.current_frame_id in self.gt_database:
            gt_data = self.gt_database[self.current_frame_id]
            boxes = gt_data['boxes']
            names = gt_data['names']
        elif view_type == 'raw' and self.current_frame_id in self.raw_predictions:
            pred = self.raw_predictions[self.current_frame_id]
            boxes = pred.get('boxes_lidar', np.array([]))
            names = pred.get('name', np.array([]))
        elif view_type == 'temporal' and self.current_frame_id in self.refined_predictions:
            pred = self.refined_predictions[self.current_frame_id]
            boxes = pred.get('boxes_lidar', np.array([]))
            names = pred.get('name', np.array([]))

        # 绘制boxes
        if len(boxes) > 0:
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            for i in range(len(boxes)):
                box = boxes[i].copy()
                box[0] += offset_x  # 平移X坐标
                
                name = names[i] if i < len(names) else 'Unknown'
                color = BOX_COLORS[view_type].get(name, [1, 1, 1])
                
                lineset = self._create_box_lineset(box, color)
                if lineset is not None:
                    geoms.append(lineset)

        return geoms

    def create_geometry_list(self):
        """创建三视图几何体"""
        geoms = []
        
        if self.points is None or len(self.points) == 0:
            return geoms

        # 计算点云范围
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)
        width = max_bound[0] - min_bound[0]
        
        # 计算偏移量
        offset_middle = width + HORIZONTAL_GAP
        offset_right = 2 * (width + HORIZONTAL_GAP)

        # 创建三个视图
        geoms.extend(self._create_single_view(0, 'gt'))           # 左: GT
        geoms.extend(self._create_single_view(offset_middle, 'raw'))  # 中: 原始
        geoms.extend(self._create_single_view(offset_right, 'temporal'))  # 右: 后处理

        # 添加坐标轴（在左侧原点）
        coord_size = max(width * 0.1, 1.0)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coord_size, origin=[0, 0, 0]
        )
        geoms.append(frame)

        return geoms

    def _count_boxes(self, view_type):
        """统计boxes数量"""
        _, names, _ = self._get_boxes_names_scores(view_type)

        counts = {}
        for name in names:
            name_norm = self._normalize_name(name)
            counts[name_norm] = counts.get(name_norm, 0) + 1
        return counts

    def _normalize_name(self, name):
        """统一类别名格式，兼容bytes/np.str_等类型。"""
        if isinstance(name, bytes):
            return name.decode('utf-8', errors='ignore')
        return str(name)

    def _get_boxes_names_scores(self, view_type):
        """获取指定视图的boxes/names/scores，缺失时返回空数组。"""
        if view_type == 'gt':
            if self.current_frame_id not in self.gt_database:
                return np.array([]), np.array([]), np.array([])
            data = self.gt_database[self.current_frame_id]
            boxes = data.get('boxes', np.array([]))
            names = data.get('names', np.array([]))
            scores = np.array([])
        elif view_type == 'raw':
            if self.current_frame_id not in self.raw_predictions:
                return np.array([]), np.array([]), np.array([])
            pred = self.raw_predictions[self.current_frame_id]
            boxes = pred.get('boxes_lidar', np.array([]))
            names = pred.get('name', np.array([]))
            # 兼容不同字段命名
            scores = pred.get('score', pred.get('scores', np.array([])))
        elif view_type == 'temporal':
            if self.current_frame_id not in self.refined_predictions:
                return np.array([]), np.array([]), np.array([])
            pred = self.refined_predictions[self.current_frame_id]
            boxes = pred.get('boxes_lidar', np.array([]))
            names = pred.get('name', np.array([]))
            scores = pred.get('score', pred.get('scores', np.array([])))
        else:
            return np.array([]), np.array([]), np.array([])

        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        names = np.array(names) if not isinstance(names, np.ndarray) else names
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores

        if boxes.size > 0 and boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        return boxes, names, scores

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*100)
        print(f"帧ID: {self.current_frame_id} | 进度: [{self.current_index+1}/{len(self.val_frames)}]")
        print("-"*100)
        
        # 点云统计
        if self.points is not None:
            print(f"点云: {len(self.points)} 点")
            labels, counts = np.unique(self.labels, return_counts=True)
            stats = []
            for l, c in zip(labels, counts):
                name = POINT_CLASS_NAMES.get(int(l), str(int(l)))
                pct = (c/len(self.points))*100
                stats.append(f"{name}:{c}({pct:.1f}%)")
            print(f"  {' | '.join(stats)}")

        print("-"*100)
        
        # 三列对比
        gt_counts = self._count_boxes('gt')
        raw_counts = self._count_boxes('raw')
        temporal_counts = self._count_boxes('temporal')

        header = f"{'GT (绿/青)':<30} | {'原始检测 (橙/品红)':<30} | {'后处理 (红/紫)':<30}"
        print(header)
        print("-"*100)

        def format_counts(counts):
            if not counts:
                return "无检测"
            parts = []
            for name, count in sorted(counts.items()):
                parts.append(f"{name}:{count}")
            return ", ".join(parts)

        gt_str = format_counts(gt_counts)
        raw_str = format_counts(raw_counts)
        temporal_str = format_counts(temporal_counts)

        print(f"{gt_str:<30} | {raw_str:<30} | {temporal_str:<30}")
        
        # 数量统计
        gt_total = sum(gt_counts.values())
        raw_total = sum(raw_counts.values())
        temporal_total = sum(temporal_counts.values())
        
        print("-"*100)
        print(f"{'总计: ' + str(gt_total):<30} | {'总计: ' + str(raw_total):<30} | {'总计: ' + str(temporal_total):<30}")
        
        # 检测变化
        if gt_total > 0:
            raw_change = raw_total - gt_total
            temporal_change = temporal_total - gt_total
            raw_symbol = "+" if raw_change > 0 else ""
            temporal_symbol = "+" if temporal_change > 0 else ""
            print(f"{'(基准)':<30} | {f'({raw_symbol}{raw_change})':<30} | {f'({temporal_symbol}{temporal_change})':<30}")

        # 逐目标明细：输出Loc/Size/Rot，预测结果额外输出score
        def print_box_details(view_label, view_type, with_score=False):
            boxes, names, scores = self._get_boxes_names_scores(view_type)
            print("-"*100)
            print(f"[{view_label}] 明细:")

            if boxes.size == 0:
                print("  无检测")
                return

            for i in range(len(boxes)):
                b = boxes[i]
                name = self._normalize_name(names[i]) if i < len(names) else 'Unknown'

                base_msg = (
                    f"  {i+1}. {name:<8} "
                    f"Loc:({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}) "
                    f"Size:({b[3]:.2f}, {b[4]:.2f}, {b[5]:.2f}) "
                    f"Rot:{b[6]:.2f}rad"
                )

                if with_score and i < len(scores):
                    score_val = float(scores[i])
                    base_msg += f" Score:{score_val:.3f}"

                print(base_msg)

        print_box_details('GT', 'gt', with_score=False)
        print_box_details('原始检测', 'raw', with_score=True)
        print_box_details('后处理检测', 'temporal', with_score=True)

        print("="*100)
        sys.stdout.flush()

    def run(self):
        """运行可视化"""
        print("\n=== 三视图检测对比可视化 ===")
        print("操作:")
        print("  ➡️  下一帧")
        print("  ⬅️  上一帧")
        print("  C   切换颜色模式 (类别/强度)")
        print("  Q   退出")
        print("="*60)
        
        if not self.load_current_frame():
            print("无法加载第一帧")
            return
            
        self.print_statistics()
        
        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="Detection Comparison (GT | Raw | Temporal)", 
                            width=1920, height=900)
            
            def safe_update(vis, step):
                try:
                    view_ctl = vis.get_view_control()
                    camera_params = view_ctl.convert_to_pinhole_camera_parameters()
                    
                    if step != 0:
                        self.current_index = (self.current_index + step) % len(self.val_frames)
                        self.load_current_frame()
                        self.print_statistics()
                    
                    vis.clear_geometries()
                    for g in self.create_geometry_list():
                        vis.add_geometry(g, reset_bounding_box=False)
                    
                    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
                except Exception as e:
                    print(f"更新视图时出错: {e}")
                    traceback.print_exc()
                return True

            # 初始化几何体
            for g in self.create_geometry_list():
                vis.add_geometry(g)
                
            # 注册按键
            vis.register_key_callback(262, lambda v: safe_update(v, 1))   # 右箭头
            vis.register_key_callback(263, lambda v: safe_update(v, -1))  # 左箭头
            
            def toggle_color(v):
                self.color_mode = "intensity" if self.color_mode == "class" else "class"
                print(f"切换颜色模式: {self.color_mode}")
                safe_update(v, 0)
                return True
                
            vis.register_key_callback(ord("C"), toggle_color)
            vis.register_key_callback(ord("Q"), lambda v: v.close())
            
            # 渲染选项
            opt = vis.get_render_option()
            opt.point_size = 2.5
            opt.background_color = np.asarray([0.95, 0.95, 0.95])  # 浅灰背景
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print("\n程序发生严重错误:")
            print(error_msg)
            with open("crash_log.txt", "w", encoding='utf-8') as f:
                f.write(error_msg)
            print("错误信息已保存至 crash_log.txt")


def main():
    try:
        app = TripleViewVisualizer()
        app.run()
    except Exception as e:
        with open("crash_log.txt", "w", encoding='utf-8') as f:
            f.write(traceback.format_exc())
        print(f"启动失败: {e}")
        print("详情请查看 crash_log.txt")


if __name__ == "__main__":
    main()
