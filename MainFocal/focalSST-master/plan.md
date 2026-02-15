## Plan: Tracklet时序后处理优化3D检测

在v4.1逐帧检测结果之上，利用原始序列的时序连续性，通过track-by-detection构建目标轨迹（tracklet），实现置信度校正、边框平滑、漏检恢复三重优化。零训练成本，纯算法后处理。

核心决策：选择后处理（而非时序网络），因为仅6个序列~3210原始帧不足以训练时序模型。使用BEV中心距离做帧间匹配（Z轴是已知瓶颈，3D IoU匹配不稳定）。train帧作为时序上下文（模型未在val帧上训练，评估公正）。

**Steps**

- [x] 1. **提取原始序列帧** — 创建 `tools/temporal_refine/extract_sequences.py`
  - 从 `data/sonar/ImageSets/train.txt` 和 `val.txt` 筛选无前缀原始帧（排除 `a_`/`d_`/`m_`/`c_`）
  - 按序列前缀（`013`-`018`）分组并按帧号排序
  - 输出 `val_original.txt`（val中的原始帧）和 `all_original.txt`（全部原始帧）
  - 输出序列映射JSON `sequence_map.json`: `{seq_id: [sorted_frame_ids]}`

- [x] 2. **批量推理所有原始帧** — 创建 `tools/temporal_refine/run_inference.py`
  - 加载v4.1 checkpoint `output/sonar_models/focal_sst_voxelnext_v4/voxelnext_v4.1/ckpt/checkpoint_epoch_24.pth`
  - 构建临时 `SonarDataset`，ImageSets指向 `all_original.txt`
  - 对每帧逐帧推理，保存predictions到 `output/temporal_refinement/raw_predictions.pkl`
  - 格式: `{frame_id: {'boxes_lidar': (N,7), 'score': (N,), 'name': (N,), 'pred_labels': (N,)}}`

- [x] 3. **实现Tracklet构建器** — 创建 `tools/temporal_refine/tracker.py`
  - 类 `SequenceTracker`，对每个序列独立运行
  - 帧间匹配：BEV中心距离 + 同类约束 + 贪心匹配
  - 参数: `match_thresh`(Box:3.0m, Diver:2.0m), `max_age`=5帧
  - 输出: 每帧检测所属tracklet编号 + tracklet元数据

- [x] 4. **实现检测优化** — 在 `tracker.py` 中实现 `refine_detections()`
  - 4a. 置信度校正: `refined_score = score × (1 + α × consistency)`, α=0.3
  - 4b. 边框平滑: 指数加权平均 w=exp(-|t-i|/τ), τ=2.0, 窗口±3帧, 不平滑heading
  - 4c. 漏检恢复: tracklet跨越但缺失的帧，线性插值box，score×0.6，min_track_len=3
  - 4d. 误检抑制: 不属于任何tracklet且score<0.3的检测移除

- [x] 5. **序列感知评估** — 创建 `tools/temporal_refine/evaluate.py`
  - 加载 `sonar_infos_val.pkl` GT，筛选仅原始val帧
  - 调用现有 `kitti_eval.get_official_eval_result()` 计算AP
  - 输出两组结果: Baseline(v4.1原始检测) vs Temporal(后处理优化)
  - 打印对比表格 (Car/Diver × 3D/BEV/bbox AP × R11/R40)

- [x] 6. **端到端Pipeline** — 创建 `tools/temporal_refine/run_pipeline.py`
  - 串联 Step 1→5，一键执行
  - 参数: `--ckpt_path`, `--cfg_file`, `--skip_inference`
  - 结果输出到 `output/temporal_refinement/`

- [x] 7. **参数调优** — 创建 `tools/temporal_refine/grid_search.py`
  - 搜索: match_thresh, α, recovery_factor, fp_thresh
  - 每组< 10秒（仅tracker+eval），可快速遍历

**Verification**
- 用序列018（158帧）单独测试tracklet构建
- 原始val帧上对比 Baseline vs Temporal 的3D AP
- 预期: Car 3D AP@0.70: 75.58→~78-80, Diver 3D AP@0.50: 58.68→~62-65
- 回归检查: BEV AP和bbox AP不下降

**Decisions**
- 后处理 vs 时序模型: 后处理（数据量不足以训练时序网络）
- BEV距离 vs 3D IoU匹配: BEV距离（Z轴噪声大）
- 贪心 vs Hungarian: 贪心（每帧~5个目标）
- heading不平滑（变化可能不连续）
- train帧作时序上下文（评估公正）