# 3D Sonar Point Cloud Visualization & Analysis Toolkit
# (3D声纳点云可视化与分析工具箱)

本项目包含一套用于处理、统计和可视化 3D 声纳点云数据（`.txt` 格式）的 Python 工具集。主要针对水下目标检测任务，支持原始数据查看、增强效果对比以及数据集批量统计。

## 📂 文件清单

| 文件名 | 功能描述 | 适用场景 |
| :--- | :--- | :--- |
| **`pointcloud_visualizer.py`** | **单视图可视化工具**。<br>支持查看单个点云文件的几何结构、包围盒尺寸及视场角参数。 | 数据清洗、单样本质量检查 |
| **`dual_pointcloud_visualizer.py`** | **双视图对比工具**。<br>左右分屏显示“原始数据”与“增强数据”，支持动态切换增强模式。 | 验证数据增强算法 (Augmentation) 的效果 |
| **`pointcloud_statistics.py`** | **数据集统计工具**。<br>批量扫描数据集，生成包含类别分布、实例中心坐标的 JSON 报告。 | 数据集概览、撰写实验报告 |

---

## ⚙️ 环境依赖

请确保你的 Python 环境已安装以下库：
```bash
pip install numpy open3d matplotlib
```
注意：本项目包含针对 Windows 长路径（\\?\）的特殊处理，建议在 Windows 10/11 环境下运行。

## 📊 数据格式说明
所有工具均默认读取 .txt 格式的点云文件。每一行代表一个点，包含 5列数据，以空格分隔：
X坐标  Y坐标  Z坐标  强度(Intensity)  类别标签(Label)

### 类别映射 (Class ID Map)
工具内部预定义的类别如下（可视化时会自动应用对应颜色）：
| ID | 类别名称 | 颜色 |
| :--- | :--- | :--- |
|1|木框 (Wood Frame)|🟤 棕色|
|2|蛙人 (Diver)|🟢 绿色|
|3|其他噪声 (Noise)|⚪ 灰色 (统计时通常排除)|
|4|水面 (Surface)|🔵 浅蓝 (统计时通常排除)|

## 🛠️ 工具详解与用法
### 1. 单视图可视化 (pointcloud_visualizer.py)
用于详细检查单个点云文件的几何属性。

**功能亮点**：
- 详细统计：控制台实时输出点数、类别分布。
- 尺寸分析：自动计算目标的 AABB 包围盒尺寸 (X/Y/Z)。
- 视场角 (FOV)：计算点云的径向距离范围、水平方位角范围和俯仰角范围。
- 快捷操作：支持键盘切换文件和着色模式。

**运行命令**：
```Bash
python pointcloud_visualizer.py --path "D:\你的数据集路径"
```

快捷键：
- ➡️ / ⬅️ : 切换上一个/下一个文件
- C : 切换着色模式（按类别着色 / 按强度热力图）
- Q : 退出

### 2. 双视图增强对比 (dual_pointcloud_visualizer.py)
专为验证数据增强算法设计。左侧显示原始数据，右侧显示增强后的数据，两者自动对齐并保持安全间距。

**文件匹配逻辑**： 脚本会自动在 path_b 中寻找与 path_a 同名的文件，支持识别以下前缀的变体：
- a_ (Affine 仿射)
- m_ (Mix 混合)
- c_ (Cutoff 截断)
- d_ (Dropout/Jitter 丢弃抖动)

**运行命令**：
```Bash
python dual_pointcloud_visualizer.py --path_a "D:\原始数据目录" --path_b "D:\增强数据目录"
```
快捷键：
- ➡️ / ⬅️ : 切换原始文件对
- M : 核心功能。在右侧视图中循环切换不同的增强变体 (如从 a_ 切到 c_)
- C : 切换着色模式

### 3. 数据集统计 (pointcloud_statistics.py)
批量遍历文件夹，生成 JSON 格式的统计报告。

**功能亮点**：
- 自动过滤：默认排除类别 3 (噪声) 和 4 (水面)，只统计有效目标。
- 实例级统计：记录每个目标的中心坐标 (XYZ) 和点数。
- 格式化输出：生成的 JSON 文件经过特殊格式化，坐标数组显示为单行，便于阅读。

**运行命令**：
```Bash
python pointcloud_statistics.py --path "D:\数据集根目录" --output "stats_result.json"
```

**输出示例** (stats_result.json):
```JSON
{
    "total_files": 100,
    "total_targets": 150,
    "class_statistics": {
        "1": {
            "name": "木框",
            "total_points": 5000,
            "instances": [
                {
                    "file_path": "part1/file1.txt",
                    "point_count": 120,
                    "center_xyz": [1.5, 2.3, -0.5]
                }
            ]
        }
    }
}
```
