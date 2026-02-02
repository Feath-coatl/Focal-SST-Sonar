import copy
import pickle
import os

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    # def get_lidar(self, idx):
    #     lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
    #     assert lidar_file.exists()
    #     point_features = np.load(lidar_file)
    #     return point_features


def get_lidar(self, idx):
    """
    核心函数：根据索引读取具体文件，添加浮点异常防护
    """
    try:
        # 设置numpy错误处理模式，将浮点错误转换为Python异常
        with np.errstate(divide='raise', over='raise', under='ignore', invalid='raise'):
            # 修改点：指向 points 子文件夹
            lidar_file = self.root_path / 'points' / f'{idx}.txt'
            
            if not lidar_file.exists():
                raise FileNotFoundError(f"File not found: {lidar_file}")
            
            print(f"DEBUG: Loading file {lidar_file}", flush=True)
            
            # 加载数据 [x, y, z, intensity, class]
            # 使用更安全的加载方式
            try:
                points_all = np.loadtxt(str(lidar_file), dtype=np.float64)  # 先用float64加载
            except ValueError as e:
                raise ValueError(f"Failed to load {idx}.txt: {e}")
            
            # 确保维度正确
            if points_all.ndim == 1:
                points_all = points_all.reshape(1, -1)
            
            if points_all.shape[0] == 0:
                raise ValueError(f"Empty point cloud in {idx}.txt")
            
            if points_all.shape[1] < 4:
                raise ValueError(f"Point cloud has {points_all.shape[1]} columns, expected at least 4")
            
            # 提取 x, y, z, intensity (前4列输入网络)
            points = points_all[:, :4].astype(np.float32)
            
            # 检查所有坐标是否有效
            if np.any(~np.isfinite(points[:, :3])):
                raise ValueError(f"Invalid coordinates (NaN/Inf) in {idx}.txt")
            
            # 提取强度值
            intensity = points[:, 3].copy()
            
            print(f"DEBUG: Intensity stats - min={intensity.min():.2e}, max={intensity.max():.2e}, mean={intensity.mean():.2e}", flush=True)
            
            # 严格检查强度值
            if np.any(~np.isfinite(intensity)):
                invalid_count = np.sum(~np.isfinite(intensity))
                print(f"WARNING: {invalid_count} invalid intensity values in {idx}, fixing...", flush=True)
                intensity = np.nan_to_num(intensity, nan=0.0, posinf=self.intensity_clip_max, neginf=0.0)
            
            # 检查是否有负值
            if np.any(intensity < 0):
                neg_count = np.sum(intensity < 0)
                print(f"WARNING: {neg_count} negative intensity values in {idx}, clipping to 0...", flush=True)
                intensity = np.maximum(intensity, 0.0)
            
            # 强度归一化 - 使用安全的方式
            # 截断离群值
            intensity = np.clip(intensity, 0.0, self.intensity_clip_max)
            
            # 对数变换 - 添加小的epsilon避免log(0)
            epsilon = 1e-10
            intensity_log = np.log10(intensity + 1.0 + epsilon)
            
            # 检查log结果
            if np.any(~np.isfinite(intensity_log)):
                print(f"ERROR: Log transform produced invalid values in {idx}", flush=True)
                raise ValueError(f"Log transform failed for {idx}")
            
            # 归一化
            if self.log_norm_divisor <= 0:
                raise ValueError(f"Invalid log_norm_divisor: {self.log_norm_divisor}")
            
            intensity_normalized = intensity_log / self.log_norm_divisor
            
            # 最终检查
            if np.any(~np.isfinite(intensity_normalized)):
                raise ValueError(f"Normalization produced invalid values in {idx}")
            
            points[:, 3] = intensity_normalized.astype(np.float32)
            
            # 坐标系对齐
            points_aligned = np.zeros_like(points, dtype=np.float32)
            points_aligned[:, 0] = points[:, 1]   # Old Y -> New X
            points_aligned[:, 1] = -points[:, 0]  # -Old X -> New Y
            points_aligned[:, 2] = points[:, 2]   # Z 保留
            points_aligned[:, 3] = points[:, 3]   # Intensity 保留
            
            # 最终检查对齐后的数据
            if np.any(~np.isfinite(points_aligned)):
                raise ValueError(f"Aligned points contain invalid values in {idx}")
            
            print(f"DEBUG: Successfully processed {idx}, points shape: {points_aligned.shape}", flush=True)
            return points_aligned
            
    except FloatingPointError as e:
        print(f"FLOATING POINT ERROR in get_lidar for {idx}: {e}", flush=True)
        raise RuntimeError(f"Floating point exception when loading {idx}: {e}")
    except Exception as e:
        print(f"ERROR in get_lidar for {idx}: {type(e).__name__}: {e}", flush=True)
        raise


    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    # def __getitem__(self, index):
    #     if self._merge_all_iters_to_one_epoch:
    #         index = index % len(self.custom_infos)

    #     info = copy.deepcopy(self.custom_infos[index])
    #     sample_idx = info['point_cloud']['lidar_idx']
    #     points = self.get_lidar(sample_idx)
    #     input_dict = {
    #         'frame_id': self.sample_id_list[index],
    #         'points': points
    #     }

    #     if 'annos' in info:
    #         annos = info['annos']
    #         annos = common_utils.drop_info_with_name(annos, name='DontCare')
    #         gt_names = annos['name']
    #         gt_boxes_lidar = annos['gt_boxes_lidar']
    #         input_dict.update({
    #             'gt_names': gt_names,
    #             'gt_boxes': gt_boxes_lidar
    #         })

    #     data_dict = self.prepare_data(data_dict=input_dict)

    #     return data_dict

    def __getitem__(self, index):
        """添加异常处理，跳过有问题的数据"""
        max_retries = 5
        original_index = index
        
        for attempt in range(max_retries):
            try:
                info = copy.deepcopy(self.sonar_infos[index])
                sample_idx = info['point_cloud']['lidar_idx']
                
                # print(f"DEBUG: Loading frame {sample_idx} (index={index}, attempt={attempt+1})", flush=True)
                
                # 读取点云
                points = self.get_lidar(sample_idx)
                
                # 检查点云是否有效
                if points is None or len(points) == 0:
                    raise ValueError(f"Empty point cloud for frame {sample_idx}")
                
                # 检查是否有NaN或Inf
                if np.isnan(points).any():
                    raise ValueError(f"NaN values in point cloud for frame {sample_idx}")
                if np.isinf(points).any():
                    raise ValueError(f"Inf values in point cloud for frame {sample_idx}")
                
                input_dict = {
                    'points': points,
                    'frame_id': sample_idx,
                }

                # 加载 GT Boxes 和 GT Names
                if 'annos' in info:
                    annos = info['annos']
                    gt_boxes = annos.get('gt_boxes_lidar', annos.get('gt_boxes', None))
                    gt_names = annos.get('name', annos.get('gt_names', None))
                    
                    if gt_boxes is not None and len(gt_boxes) > 0:
                        input_dict['gt_boxes'] = gt_boxes
                        input_dict['gt_names'] = gt_names
                
                # 调用 prepare_data 进行数据增强等处理
                data_dict = self.prepare_data(data_dict=input_dict)
                
                print(f"DEBUG: Successfully loaded frame {sample_idx}", flush=True)
                return data_dict
                
            except Exception as e:
                error_msg = f"Error loading sample {index} (frame {sample_idx if 'sample_idx' in locals() else 'unknown'}): {type(e).__name__}: {e}"
                print(f"DEBUG ERROR: {error_msg}", flush=True)
                
                # 跳到下一个索引
                index = (index + 1) % len(self.sonar_infos)
                print(f"DEBUG: Skipping to next index {index}...", flush=True)
                
                if index == original_index:
                    # 已经循环一圈了，所有数据都有问题
                    raise RuntimeError(f"All samples appear to be corrupted, starting from index {original_index}")
        
        # 如果所有重试都失败了
        raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries starting from index {original_index}")

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is save to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is save to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )
