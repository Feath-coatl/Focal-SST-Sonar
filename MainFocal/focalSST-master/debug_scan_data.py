import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# === 配置 ===
ROOT_PATH = Path('data/sonar')
POINTS_DIR = ROOT_PATH / 'points'
INFO_FILE = ROOT_PATH / 'sonar_infos_train.pkl'
# ===========

def check_data():
    print(f"正在扫描点云文件: {POINTS_DIR}")
    
    # 1. 扫描所有 TXT 文件
    txt_files = list(POINTS_DIR.glob('*.txt'))
    print(f"找到 {len(txt_files)} 个文件")
    
    error_files = []
    
    for txt_file in tqdm(txt_files):
        try:
            # 读取
            data = np.loadtxt(str(txt_file), dtype=np.float32)
            
            # 检查1: 空文件
            if data.size == 0:
                print(f"\n[ERROR] 文件为空: {txt_file.name}")
                error_files.append(txt_file.name)
                continue
                
            if data.ndim == 1: data = data.reshape(1, -1)
            
            # 检查2: NaN 或 Inf (致命杀手)
            if np.isnan(data).any():
                print(f"\n[FATAL] 发现 NaN: {txt_file.name}")
                error_files.append(txt_file.name)
                continue
                
            if np.isinf(data).any():
                print(f"\n[FATAL] 发现 Inf: {txt_file.name}")
                error_files.append(txt_file.name)
                continue
                
            # 检查3: 强度异常 (Log处理前的负数会导致NaN)
            # 你的处理是 log1p(intensity)，所以 intensity 不能 <= -1
            if (data[:, 3] <= -1).any():
                print(f"\n[FATAL] 强度值异常(<= -1): {txt_file.name}")
                error_files.append(txt_file.name)
                
        except Exception as e:
            print(f"\n[ERROR] 读取失败 {txt_file.name}: {e}")
            error_files.append(txt_file.name)

    print("-" * 30)
    
    # 2. 再次深度扫描 PKL (防止上一轮修复不彻底)
    print(f"正在复查 PKL 文件: {INFO_FILE}")
    with open(INFO_FILE, 'rb') as f:
        infos = pickle.load(f)
        
    for i, info in enumerate(infos):
        if 'annos' in info and 'gt_boxes_lidar' in info['annos']:
            boxes = info['annos']['gt_boxes_lidar']
            if len(boxes) > 0:
                # 检查 Box 是否有 NaN
                if np.isnan(boxes).any() or np.isinf(boxes).any():
                    print(f"\n[FATAL] GT Box 包含 NaN/Inf! Index: {i}, Lidar ID: {info['point_cloud']['lidar_idx']}")
                
                # 检查 Box 体积是否过小 (防止除以零)
                # dx, dy, dz 分别是 index 3, 4, 5
                vols = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]
                if (vols < 1e-6).any():
                    print(f"\n[FATAL] GT Box 体积接近 0! Index: {i}, Lidar ID: {info['point_cloud']['lidar_idx']}")
                    print(f"Boxes: {boxes}")

    if len(error_files) == 0:
        print("\n✅ 数据扫描完成，未发现显式 NaN/Inf/空文件。")
    else:
        print(f"\n❌ 发现 {len(error_files)} 个问题文件，请立即删除或修复！")

if __name__ == '__main__':
    check_data()