import numpy as np
from ...utils import box_utils


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        # 关键修复：用列表推导创建新数组，避免numpy dtype截断
        # 原始name数组可能是<U5（如'Box','Diver'），赋值'Pedestrian'(10字符)
        # 会被numpy静默截断为'Pedes'，导致KITTI eval匹配失败 → AP=0
        anno['name'] = np.array([map_name_to_kitti[n] for n in anno['name']])

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2