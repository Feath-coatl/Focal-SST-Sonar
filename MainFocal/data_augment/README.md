# 3D Sonar Point Cloud Data Augmentation Toolkit
# (3D声纳点云数据增强工具箱)

本项目包含一套针对 3D 声纳点云数据（`.txt` 格式）的离线数据增强流水线。通过构建前景目标数据库，支持对原始数据进行仿射变换、混合增强（Mix）、边界截断（Cutoff）以及噪声抖动（Dropout/Jitter）处理，旨在扩充训练数据集并提升模型鲁棒性。

## 📂 文件清单

| 文件名 | 功能描述 | 核心逻辑 |
| :--- | :--- | :--- |
| **`build_object_db.py`** | **目标库构建工具**。<br>扫描数据集，提取前景目标（如蛙人、铁框等），归一化后保存为 pickle 数据库文件。 | 提取 Class 1,2,5,6,7,8 -> 去中心化 -> 计算相对水面高度 -> 存入 `.pkl` |
| **`run_augmentation.py`** | **增强主程序**。<br>加载目标库，根据指定模式（Mode）对源数据进行批量增强并保存。 | 包含 Affine, Mix, Cutoff, Dropout 四种模式的调度逻辑。 |
| **`augment_utils.py`** | **底层算法库**。<br>提供几何变换、物理校验、文件IO等通用函数。 | 坐标转换 (Cart/Polar)、碰撞检测、FOV校验、高斯抖动、按强度Dropout等。 |

---

## ⚙️ 环境依赖

请确保你的 Python 环境已安装以下库：
```bash
pip install numpy scikit-learn tqdm
```

## 使用流程
### 第一步：构建目标数据库 (Build Object Database)
在运行增强之前，必须先提取现有数据集中的前景目标，建立实例库。

**操作说明**： 打开 build_object_db.py，修改底部的 DATA_PATH 为你的数据集根目录，然后运行：

```Bash
python build_object_db.py
```
- 输入：原始 .txt 数据集文件夹。
- 输出：object_db.pkl (包含归一化后的目标点云及其元数据)。
- 提取类别：默认提取 ID 为 [1, 2, 5, 6, 7, 8] 的目标。

### 第二步：运行数据增强 (Run Augmentation)
使用 run_augmentation.py 对数据集进行批量增强。

**参数说明**：
- --src : 源数据根目录 (Source)。
- --dst : 增强结果保存目录 (Destination)。
- --db : 目标数据库路径 (默认 object_db.pkl)。
- --mode : 增强模式，可选 affine, mix, cutoff, dropout, all (默认)。

运行示例：
执行所有增强模式 (推荐)：
```Bash
python run_augmentation.py --src "D:\Datasets\Origin" --dst "D:\Datasets\Augmented" --mode all
```
仅执行混合增强 (Mix Mode)：
```Bash
python run_augmentation.py --src "D:\Datasets\Origin" --dst "D:\Datasets\Augmented" --mode mix
```

## 🎨 增强模式详解 (Augmentation Modes)
工具会自动根据模式为生成的文件添加前缀，方便区分。

模式|前缀|算法逻辑|适用场景
| :--- | :--- | :--- | :--- |
Affine|a_|仿射变换。保持场景背景不变，将前景目标移动到新的随机合法位置（需通过碰撞检测和 FOV 校验），并添加背景噪声。|模拟目标运动、不同视角下的形态。
Mix|m_|混合增强。从 object_db.pkl 中随机抽取一个额外目标，注入到当前场景的空白区域中。|解决样本不平衡，增加多目标场景。
Cutoff|c_|边界截断。模拟声纳扫描边缘。尝试将目标移动到 FOV 边界或水面附近，使其部分点云被截断（丢失）。注：若无法生成有效的截断效果，该帧将不会被保存。|提升模型对残缺目标的识别能力。
Dropout|d_|丢弃与抖动。原地对目标进行随机旋转/缩放，并基于反射强度进行随机丢点（Dropout 5%-15%），同时叠加高斯噪声（Jitter）。|模拟信号衰减、传感器噪声。

## 📊 数据格式与类别
本工具处理标准的 .txt 点云文件（X Y Z Intensity Label）。

**处理逻辑中的类别定义**：
- 前景目标 (Foreground): ID [1, 2] (用于构建库和变换)。
- 背景 (Background): ID 3 (在增强时作为环境保留，并从中提取真实噪声模板)。
- 水面 (Water): ID 4 (用于计算吃水深度和高度限制)。

## ⚠️ 注意事项
**文件覆盖**：程序会自动保持源目录的文件夹结构。
**长路径问题**：在 Windows 下如果路径过长，建议开启长路径支持或使用较短的根目录。
**Cutoff 模式**：该模式具有筛选机制，只有当目标真正产生截断效果（剩余点比例 < 95%）时才会保存文件，因此生成的文件数量可能少于源文件数量。