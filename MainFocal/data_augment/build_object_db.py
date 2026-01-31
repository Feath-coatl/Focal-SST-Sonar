import os
import numpy as np
import pickle
import glob
from tqdm import tqdm 
import augment_utils as utils

def build_database(dataset_path, output_db_path="object_db.pkl"):
    FOREGROUND_CLASSES = [1, 2]
    object_db = {k: [] for k in FOREGROUND_CLASSES}
    
    file_list = glob.glob(os.path.join(dataset_path, "**/*.txt"), recursive=True)
    print(f"[Info] 找到 {len(file_list)} 个文件，开始构建数据库...")
    
    count = 0
    
    for fpath in tqdm(file_list):
        data = utils.read_txt(fpath)
        if data is None: continue
        
        labels = data[:, 4].astype(int)
        water_level = utils.get_water_level(data)
        if water_level is None:
            water_level = np.min(data[:, 2])

        for cls_id in FOREGROUND_CLASSES:
            cls_mask = labels == cls_id
            if np.sum(cls_mask) == 0: continue
                
            cls_points = data[cls_mask]
            
            # [修改点] 移除 DBSCAN 聚类。
            # 假设：单帧中每类目标只有一个实例，直接使用 cls_points 作为实例数据。
            # 为了兼容原有结构，将其放入列表中
            clusters = [cls_points]
            
            for instance_data in clusters:
                xyz = instance_data[:, :3]
                center = np.mean(xyz, axis=0)
                
                normalized_xyz = xyz - center
                instance_data[:, :3] = normalized_xyz
                
                # 计算吃水深度/悬浮高度
                z_offset = np.min(xyz[:, 2]) - water_level
                
                object_db[cls_id].append({
                    "points": instance_data,   # Normalized (N, 5)
                    "z_offset": z_offset,
                    "class_id": cls_id
                })
                count += 1
                
    print(f"[Success] 数据库构建完成，共提取 {count} 个实例。")
    with open(output_db_path, "wb") as f:
        pickle.dump(object_db, f)
    print(f"[Saved] 数据库已保存至: {output_db_path}")

if __name__ == "__main__":
    DATA_PATH = r"d:\BaiduNetdiskDownload\datasets" 
    build_database(DATA_PATH)