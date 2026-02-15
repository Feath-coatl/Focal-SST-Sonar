"""
Step 3 & 4: Tracklet构建器 + 检测优化
对每个序列独立运行track-by-detection，构建目标轨迹。
实现置信度校正、边框平滑、漏检恢复、误检抑制四重优化。

核心设计:
  - 帧间匹配: BEV中心距离 + 同类约束 + 贪心匹配
  - 参数: match_thresh(Box:3.0m, Diver:2.0m), max_age=5帧
"""

import numpy as np
from collections import defaultdict
from copy import deepcopy


class Track:
    """单个目标的轨迹(Tracklet)"""
    
    def __init__(self, track_id, frame_idx, det_idx, box, score, label, name):
        self.track_id = track_id
        self.history = []  # [(frame_idx, det_idx, box, score)]
        self.label = label
        self.name = name
        self.age = 0  # 连续未匹配帧数
        self.total_hits = 0
        
        self.add_detection(frame_idx, det_idx, box, score)
    
    def add_detection(self, frame_idx, det_idx, box, score):
        """添加一个新的检测匹配"""
        self.history.append({
            'frame_idx': frame_idx,
            'det_idx': det_idx,
            'box': box.copy(),
            'score': score,
        })
        self.age = 0
        self.total_hits += 1
    
    def increment_age(self):
        self.age += 1
    
    @property
    def last_box(self):
        return self.history[-1]['box']
    
    @property
    def last_score(self):
        return self.history[-1]['score']
    
    @property
    def last_frame_idx(self):
        return self.history[-1]['frame_idx']
    
    @property
    def frame_span(self):
        """轨迹跨越的帧范围"""
        if len(self.history) == 0:
            return 0
        return self.history[-1]['frame_idx'] - self.history[0]['frame_idx'] + 1
    
    @property
    def avg_score(self):
        """轨迹的平均置信度"""
        if len(self.history) == 0:
            return 0.0
        return sum(h['score'] for h in self.history) / len(self.history)


class SequenceTracker:
    """
    单序列Tracklet构建器
    使用BEV中心距离做帧间匹配的贪心算法
    """
    
    def __init__(self, config=None):
        self.config = config or self.default_config()
        self.tracks = []
        self.next_track_id = 0
        self.finished_tracks = []
    
    @staticmethod
    def default_config():
        return {
            # 匹配阈值（BEV距离，米）— 基于数据集先验优化
            'match_thresh': {
                'Box': 0.8,    # Box极慢移动（两帧<1m），严格阈值
                'Diver': 1.5,
            },
            'default_match_thresh': 2.5,
            # 最大未匹配年龄
            'max_age': 5,
            # ========== 检测优化参数 ==========
            # 置信度校正
            'alpha': 0.4,           # refined_score = score * (1 + alpha * consistency)
            # 边框平滑
            'smooth_tau': 2.0,      # 指数加权平均的时间常数
            'smooth_window': 3,     # 平滑窗口 ±N 帧
            'smooth_heading': False, # 不平滑heading
            # 漏检恢复
            'min_track_len': 5,     # 最短轨迹长度（才执行恢复）
            'recovery_score_factor': 0.4,  # 恢复帧的score缩放因子
            'min_track_score': 0.5, # 轨迹平均分数阈值（高于此值才恢复）
            'max_recovery_gap': 3,  # 最大恢复间隔（帧）
            # 误检抑制
            'fp_score_thresh': 0.35, # 不属于任何tracklet且score<此值的检测被移除
            # ========== 数据集先验约束 ==========
            'per_class_max_objects': {'Box': 1, 'Diver': 1},  # 每帧每类最多目标数
            'enforce_single_track_per_class': True,  # 每序列每类最多1条活跃轨迹
        }
    
    def _get_match_thresh(self, name):
        return self.config['match_thresh'].get(name, self.config['default_match_thresh'])
    
    def _bev_distance(self, box1, box2):
        """计算两个box的BEV中心距离"""
        return np.sqrt((box1[0] - box2[0])**2 + (box1[1] - box2[1])**2)
    
    def _greedy_match(self, detections, active_tracks):
        """
        贪心匹配: 按距离排序，逐一匹配
        
        Args:
            detections: list of (det_idx, box, score, label, name)
            active_tracks: list of Track objects
        
        Returns:
            matches: list of (track_idx, det_idx)
            unmatched_dets: list of det_idx
            unmatched_tracks: list of track_idx
        """
        if len(detections) == 0 or len(active_tracks) == 0:
            return [], list(range(len(detections))), list(range(len(active_tracks)))
        
        # 计算距离矩阵
        cost_matrix = np.full((len(active_tracks), len(detections)), 1e6)
        
        for t_idx, track in enumerate(active_tracks):
            for d_idx, det in enumerate(detections):
                det_idx, box, score, label, name = det
                # 同类约束
                if name != track.name:
                    continue
                dist = self._bev_distance(track.last_box, box)
                thresh = self._get_match_thresh(name)
                if dist < thresh:
                    cost_matrix[t_idx, d_idx] = dist
        
        # 贪心匹配: 按距离从小到大
        matches = []
        matched_tracks = set()
        matched_dets = set()
        
        # 获取所有有效配对并排序
        pairs = []
        for t_idx in range(len(active_tracks)):
            for d_idx in range(len(detections)):
                if cost_matrix[t_idx, d_idx] < 1e5:
                    pairs.append((cost_matrix[t_idx, d_idx], t_idx, d_idx))
        
        pairs.sort(key=lambda x: x[0])
        
        for dist, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matches.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
        
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_tracks = [t for t in range(len(active_tracks)) if t not in matched_tracks]
        
        return matches, unmatched_dets, unmatched_tracks
    
    def process_sequence(self, frame_ids, predictions):
        """
        处理单个序列的所有帧
        
        Args:
            frame_ids: 排序后的帧ID列表
            predictions: {frame_id: {'boxes_lidar': (N,7), 'score': (N,), 
                                      'name': (N,), 'pred_labels': (N,)}}
        
        Returns:
            tracking_results: {frame_id: list of {
                'det_idx': int, 'track_id': int, 'box': array, 'score': float, 
                'name': str, 'label': int
            }}
            all_tracks: list of Track objects (所有已完成的轨迹)
        """
        self.tracks = []
        self.finished_tracks = []
        self.next_track_id = 0
        
        tracking_results = {}
        
        for f_idx, frame_id in enumerate(frame_ids):
            pred = predictions.get(frame_id, None)
            
            if pred is None or len(pred['score']) == 0:
                # 空帧 — 增加所有活跃track的年龄
                for track in self.tracks:
                    track.increment_age()
                self._retire_old_tracks()
                tracking_results[frame_id] = []
                continue
            
            # 构建检测列表
            detections = []
            for d_idx in range(len(pred['score'])):
                detections.append((
                    d_idx,
                    pred['boxes_lidar'][d_idx],
                    pred['score'][d_idx],
                    pred['pred_labels'][d_idx],
                    pred['name'][d_idx],
                ))
            
            # 贪心匹配
            matches, unmatched_dets, unmatched_tracks = self._greedy_match(
                detections, self.tracks
            )
            
            frame_results = []
            
            # 更新匹配的track
            for t_idx, d_idx in matches:
                det = detections[d_idx]
                self.tracks[t_idx].add_detection(f_idx, det[0], det[1], det[2])
                frame_results.append({
                    'det_idx': det[0],
                    'track_id': self.tracks[t_idx].track_id,
                    'box': det[1].copy(),
                    'score': det[2],
                    'name': det[4],
                    'label': det[3],
                })
            
            # 为未匹配的检测创建新track（应用轨迹级约束）
            for d_idx in unmatched_dets:
                det = detections[d_idx]
                det_name = det[4]
                
                # 数据集先验约束：每类最多1条活跃轨迹
                if self.config.get('enforce_single_track_per_class', False):
                    # 检查是否已有同类活跃轨迹
                    has_active_track = any(t.name == det_name for t in self.tracks)
                    if has_active_track:
                        # 跳过此检测，避免同类多轨迹
                        continue
                
                new_track = Track(
                    track_id=self.next_track_id,
                    frame_idx=f_idx,
                    det_idx=det[0],
                    box=det[1],
                    score=det[2],
                    label=det[3],
                    name=det[4],
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
                
                frame_results.append({
                    'det_idx': det[0],
                    'track_id': new_track.track_id,
                    'box': det[1].copy(),
                    'score': det[2],
                    'name': det[4],
                    'label': det[3],
                })
            
            # 增加未匹配track的年龄
            for t_idx in unmatched_tracks:
                self.tracks[t_idx].increment_age()
            
            self._retire_old_tracks()
            tracking_results[frame_id] = frame_results
        
        # 将所有剩余活跃track归档
        self.finished_tracks.extend(self.tracks)
        self.tracks = []
        
        return tracking_results, self.finished_tracks
    
    def _retire_old_tracks(self):
        """退役超龄的track"""
        still_active = []
        for track in self.tracks:
            if track.age > self.config['max_age']:
                self.finished_tracks.append(track)
            else:
                still_active.append(track)
        self.tracks = still_active


def refine_detections(frame_ids, predictions, tracking_results, all_tracks, config=None):
    """
    Step 4: 检测优化
    基于tracklet信息对逐帧检测结果进行四重优化:
    4a. 置信度校正
    4b. 边框平滑
    4c. 漏检恢复
    4d. 误检抑制

    Args:
        frame_ids: 排序后的帧ID列表
        predictions: 原始检测结果 {frame_id: {...}}
        tracking_results: tracker输出的帧级tracking结果
        all_tracks: 所有Track对象
        config: 参数配置

    Returns:
        refined_predictions: 优化后的检测结果 {frame_id: {'boxes_lidar': ..., 'score': ..., 'name': ..., 'pred_labels': ...}}
    """
    if config is None:
        config = SequenceTracker.default_config()
    
    alpha = config['alpha']
    smooth_tau = config['smooth_tau']
    smooth_window = config['smooth_window']
    smooth_heading = config['smooth_heading']
    min_track_len = config['min_track_len']
    recovery_score_factor = config['recovery_score_factor']
    fp_score_thresh = config['fp_score_thresh']
    min_track_score = config.get('min_track_score', 0.5)
    max_recovery_gap = config.get('max_recovery_gap', 3)

    # 建立 frame_id → frame_idx 映射
    frame_id_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}
    
    # 建立 track_id → Track 映射
    track_map = {t.track_id: t for t in all_tracks}
    
    # 建立 (frame_id, det_idx) → track_id 映射
    det_to_track = {}
    for frame_id, results in tracking_results.items():
        for r in results:
            det_to_track[(frame_id, r['det_idx'])] = r['track_id']
    
    # ========= 4a. 置信度校正 =========
    # consistency = track_hits / track_frame_span
    track_consistency = {}
    for track in all_tracks:
        if track.frame_span > 0:
            consistency = track.total_hits / track.frame_span
        else:
            consistency = 0.0
        track_consistency[track.track_id] = consistency
    
    # ========= 初始化 refined_predictions =========
    refined_predictions = {}
    
    for frame_id in frame_ids:
        pred = predictions.get(frame_id, None)
        if pred is None or len(pred['score']) == 0:
            refined_predictions[frame_id] = {
                'boxes_lidar': np.zeros((0, 7), dtype=np.float32),
                'score': np.zeros(0, dtype=np.float32),
                'name': np.array([], dtype='<U10'),
                'pred_labels': np.zeros(0, dtype=np.int32),
            }
            continue
        
        n_dets = len(pred['score'])
        new_boxes = pred['boxes_lidar'].copy()
        new_scores = pred['score'].copy()
        new_names = pred['name'].copy()
        new_labels = pred['pred_labels'].copy()
        
        keep_mask = np.ones(n_dets, dtype=bool)
        
        for d_idx in range(n_dets):
            key = (frame_id, d_idx)
            
            if key in det_to_track:
                track_id = det_to_track[key]
                track = track_map[track_id]
                
                # 4a. 置信度校正
                consistency = track_consistency[track_id]
                new_scores[d_idx] = new_scores[d_idx] * (1 + alpha * consistency)
                new_scores[d_idx] = min(new_scores[d_idx], 1.0)  # 截断到1.0
                
                # 4b. 边框平滑 (仅对足够长的tracklet)
                if track.total_hits >= min_track_len:
                    f_idx = frame_id_to_idx[frame_id]
                    smoothed_box = _smooth_box(
                        track, f_idx, smooth_tau, smooth_window, smooth_heading
                    )
                    if smoothed_box is not None:
                        new_boxes[d_idx] = smoothed_box
            else:
                # 4d. 误检抑制: 不属于任何tracklet且score < fp_thresh
                if new_scores[d_idx] < fp_score_thresh:
                    keep_mask[d_idx] = False
        
        # 应用掩码
        refined_predictions[frame_id] = {
            'boxes_lidar': new_boxes[keep_mask],
            'score': new_scores[keep_mask],
            'name': new_names[keep_mask],
            'pred_labels': new_labels[keep_mask],
        }
    
    # ========= 4c. 漏检恢复 =========
    recovered_count = _recover_missing_detections(
        frame_ids, refined_predictions, all_tracks, 
        frame_id_to_idx, min_track_len, recovery_score_factor,
        min_track_score, max_recovery_gap
    )
    
    # ========= 4e. 数据集先验约束：单目标约束 =========
    # 每帧每类只保留置信度最高的1个检测
    per_class_max = config.get('per_class_max_objects', {})
    uniqueness_removed = 0
    if per_class_max:
        uniqueness_removed = _apply_per_class_uniqueness(refined_predictions, per_class_max)
    
    return refined_predictions, recovered_count, uniqueness_removed


def _smooth_box(track, target_frame_idx, tau, window, smooth_heading):
    """
    对tracklet中某一帧的box做指数加权平均平滑
    
    Args:
        track: Track对象
        target_frame_idx: 当前帧在序列中的索引
        tau: 指数衰减常数
        window: 窗口半径 ±N帧
        smooth_heading: 是否平滑heading角
    
    Returns:
        smoothed_box: (7,) 或 None
    """
    # 收集窗口内的box
    boxes = []
    weights = []
    
    for h in track.history:
        dt = abs(h['frame_idx'] - target_frame_idx)
        if dt <= window:
            w = np.exp(-dt / tau)
            boxes.append(h['box'])
            weights.append(w)
    
    if len(boxes) <= 1:
        return None  # 只有自己，不需要平滑
    
    boxes = np.array(boxes)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # 加权平均 (除heading外)
    smoothed = np.zeros(7, dtype=np.float32)
    
    if smooth_heading:
        # 全部7维加权平均 (heading需要特殊处理角度环绕)
        smoothed = np.average(boxes, axis=0, weights=weights)
    else:
        # 前6维: x, y, z, dx, dy, dz
        smoothed[:6] = np.average(boxes[:, :6], axis=0, weights=weights)
        # heading保持原值 — 找到target_frame_idx对应的entry
        for h in track.history:
            if h['frame_idx'] == target_frame_idx:
                smoothed[6] = h['box'][6]
                break
        else:
            # 如果没找到精确匹配，使用最近的
            smoothed[6] = boxes[np.argmax(weights), 6]
    
    return smoothed


def _recover_missing_detections(frame_ids, refined_predictions, all_tracks,
                                 frame_id_to_idx, min_track_len, recovery_score_factor,
                                 min_track_score=0.5, max_recovery_gap=3):
    """
    4c. 漏检恢复: 对于tracklet跨越但缺失的帧，线性插值box
    
    修改 refined_predictions in-place
    
    添加质量控制：
    - 只对高质量轨迹恢复（avg_score >= min_track_score）
    - 只恢复短gap（<= max_recovery_gap帧）
    """
    recovered_count = 0
    for track in all_tracks:
        if track.total_hits < min_track_len:
            continue
        
        # 质量阈值：只对高置信度轨迹恢复
        if track.avg_score < min_track_score:
            continue
        
        history = track.history
        first_f_idx = history[0]['frame_idx']
        last_f_idx = history[-1]['frame_idx']
        
        # 收集tracklet覆盖的所有帧索引
        covered_f_indices = set(h['frame_idx'] for h in history)
        
        # 找出缺失的帧
        for f_idx in range(first_f_idx, last_f_idx + 1):
            if f_idx in covered_f_indices:
                continue
            
            # 这个帧缺失了 — 需要插值
            if f_idx < 0 or f_idx >= len(frame_ids):
                continue
            
            frame_id = frame_ids[f_idx]
            
            # 找前后最近的检测
            prev_h = None
            next_h = None
            for h in history:
                if h['frame_idx'] < f_idx:
                    prev_h = h
                elif h['frame_idx'] > f_idx and next_h is None:
                    next_h = h
            
            if prev_h is None or next_h is None:
                continue
            
            # 限制gap大小：只恢复短距离缺失
            gap = next_h['frame_idx'] - prev_h['frame_idx']
            if gap > max_recovery_gap:
                continue
            
            # 线性插值box
            t = (f_idx - prev_h['frame_idx']) / (next_h['frame_idx'] - prev_h['frame_idx'])
            interp_box = prev_h['box'] * (1 - t) + next_h['box'] * t
            
            # heading角需要特殊处理（短弧插值）
            h1, h2 = prev_h['box'][6], next_h['box'][6]
            dh = h2 - h1
            # 归一化到 [-pi, pi]
            while dh > np.pi: dh -= 2 * np.pi
            while dh < -np.pi: dh += 2 * np.pi
            interp_box[6] = h1 + t * dh
            
            # score = 前后均值 × recovery_factor
            interp_score = (prev_h['score'] + next_h['score']) / 2 * recovery_score_factor
            
            # 添加到refined_predictions
            pred = refined_predictions[frame_id]
            pred['boxes_lidar'] = np.vstack([
                pred['boxes_lidar'], 
                interp_box.reshape(1, -1)
            ]) if len(pred['boxes_lidar']) > 0 else interp_box.reshape(1, -1)
            pred['score'] = np.append(pred['score'], interp_score)
            pred['name'] = np.append(pred['name'], track.name)
            pred['pred_labels'] = np.append(pred['pred_labels'], track.label)
            recovered_count += 1
    
    return recovered_count


def _apply_per_class_uniqueness(refined_predictions, per_class_max):
    """
    4e. 数据集先验约束：每帧每类只保留置信度最高的N个检测
    
    基于数据集先验：每帧最多1个Diver + 1个Box
    修改 refined_predictions in-place
    
    Args:
        refined_predictions: {frame_id: {'boxes_lidar': ..., 'score': ..., 'name': ..., 'pred_labels': ...}}
        per_class_max: {'Box': 1, 'Diver': 1} 每类最大保留数量
    
    Returns:
        removed_count: 被移除的检测数量
    """
    removed_count = 0
    
    for frame_id, pred in refined_predictions.items():
        if len(pred['score']) == 0:
            continue
        
        # 按类别分组
        class_indices = defaultdict(list)
        for i, name in enumerate(pred['name']):
            class_indices[name].append(i)
        
        # 对每个类别，只保留top-K
        keep_indices = []
        for class_name, indices in class_indices.items():
            max_keep = per_class_max.get(class_name, len(indices))  # 默认保留全部
            
            if len(indices) <= max_keep:
                keep_indices.extend(indices)
            else:
                # 按score降序排序，保留top-K
                indices_sorted = sorted(indices, key=lambda i: pred['score'][i], reverse=True)
                keep_indices.extend(indices_sorted[:max_keep])
                removed_count += len(indices) - max_keep
        
        # 应用保留掩码
        if len(keep_indices) < len(pred['score']):
            keep_indices = sorted(keep_indices)  # 保持原始顺序
            refined_predictions[frame_id] = {
                'boxes_lidar': pred['boxes_lidar'][keep_indices],
                'score': pred['score'][keep_indices],
                'name': pred['name'][keep_indices],
                'pred_labels': pred['pred_labels'][keep_indices],
            }
    
    return removed_count


def run_tracking_and_refinement(sequence_map, predictions, config=None):
    """
    对所有序列执行tracking + refinement的完整流程
    
    Args:
        sequence_map: {seq_id: [sorted_frame_ids]}
        predictions: {frame_id: {'boxes_lidar': ..., 'score': ..., 'name': ..., 'pred_labels': ...}}
        config: 参数配置
    
    Returns:
        all_refined: {frame_id: {...}} 所有帧的优化后检测结果
        all_tracks_by_seq: {seq_id: [Track, ...]}
        stats: 统计信息dict
    """
    if config is None:
        config = SequenceTracker.default_config()
    
    all_refined = {}
    all_tracks_by_seq = {}
    
    stats = {
        'total_original_dets': 0,
        'total_refined_dets': 0,
        'total_tracks': 0,
        'total_recovered': 0,
        'total_suppressed': 0,
        'total_uniqueness_removed': 0,
    }
    
    for seq_id, frame_ids in sorted(sequence_map.items()):
        print(f"\n处理序列 {seq_id} ({len(frame_ids)} 帧)...")
        
        # 统计原始检测数
        orig_dets = sum(
            len(predictions.get(fid, {}).get('score', []))
            for fid in frame_ids
        )
        stats['total_original_dets'] += orig_dets
        
        # 1. Tracking
        tracker = SequenceTracker(config)
        tracking_results, all_tracks = tracker.process_sequence(frame_ids, predictions)
        all_tracks_by_seq[seq_id] = all_tracks
        stats['total_tracks'] += len(all_tracks)
        
        print(f"  原始检测: {orig_dets}, 轨迹数: {len(all_tracks)}")
        
        # 轨迹长度统计
        track_lens = [t.total_hits for t in all_tracks]
        if track_lens:
            print(f"  轨迹长度: min={min(track_lens)}, max={max(track_lens)}, "
                  f"avg={sum(track_lens)/len(track_lens):.1f}")
        
        # 2. Refinement
        refined, recovered_count, uniqueness_removed = refine_detections(
            frame_ids, predictions, tracking_results, all_tracks, config
        )
        
        # 统计优化后检测数
        refined_dets = sum(len(v['score']) for v in refined.values())
        stats['total_refined_dets'] += refined_dets
        
        delta = refined_dets - orig_dets
        suppressed_count = max(0, -delta + recovered_count - uniqueness_removed)
        stats['total_recovered'] += recovered_count
        stats['total_suppressed'] += suppressed_count
        stats['total_uniqueness_removed'] = stats.get('total_uniqueness_removed', 0) + uniqueness_removed
        
        print(f"  优化后检测: {refined_dets} (恢复+{recovered_count}, 误检抑制-{suppressed_count}, 单目标约束-{uniqueness_removed})")
        
        all_refined.update(refined)
    
    print(f"\n{'='*50}")
    print(f"总计: 原始 {stats['total_original_dets']} → 优化 {stats['total_refined_dets']} 检测")
    print(f"  轨迹: {stats['total_tracks']}")
    print(f"  恢复: +{stats['total_recovered']}")
    print(f"  误检抑制: -{stats['total_suppressed']}")
    print(f"  单目标约束: -{stats['total_uniqueness_removed']}")
    
    return all_refined, all_tracks_by_seq, stats
