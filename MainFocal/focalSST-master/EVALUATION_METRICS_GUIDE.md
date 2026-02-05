# 评估指标说明文档

## 📊 训练后自动评估的指标含义

训练完成后，程序会自动在验证集上进行评估，输出的指标如下：

### 1. Recall指标（召回率）

#### recall_roi_X 和 recall_rcnn_X

```
recall_roi_0.3: 0.000000
recall_rcnn_0.3: 0.835660
recall_roi_0.5: 0.000000
recall_rcnn_0.5: 0.652352
recall_roi_0.7: 0.000000
recall_rcnn_0.7: 0.387253
```

**含义解释**：

- **recall_roi_X**: Region of Interest (候选区域)的召回率
  - `roi_0.3`：IoU阈值≥0.3时，RoI生成阶段的召回率
  - 你的结果显示为0.000000，说明模型可能**没有RoI阶段**（CenterPoint是anchor-free单阶段检测器）

- **recall_rcnn_X**: 最终检测结果的召回率
  - `rcnn_0.3`：IoU阈值≥0.3时，最终检测的召回率为**83.57%**
  - `rcnn_0.5`：IoU阈值≥0.5时，召回率为**65.24%**
  - `rcnn_0.7`：IoU阈值≥0.7时，召回率为**38.73%**

**IoU阈值含义**：
- IoU (Intersection over Union)：预测框与真实框的重叠度
- IoU ≥ 0.3：宽松标准（30%重叠即算正确）
- IoU ≥ 0.5：中等标准（50%重叠）
- IoU ≥ 0.7：严格标准（70%重叠）

**你的模型表现**：
- ✅ 宽松标准(0.3)下召回率很高：83.57%
- ⚠️ 严格标准(0.7)下召回率较低：38.73%
- 说明：模型能检测到大部分目标，但**定位精度**还有提升空间

---

### 2. RECALL_THRESH_LIST 配置

在你的 `focal_sst.yaml` 中：

```yaml
POST_PROCESSING:
    RECALL_THRESH_LIST: [0.3, 0.5, 0.7]  # 评估时使用的IoU阈值列表
    SCORE_THRESH: 0.1                    # 检测置信度阈值
    OUTPUT_RAW_SCORE: False
```

**作用**：
- `RECALL_THRESH_LIST`：定义评估时计算召回率的IoU阈值
- `[0.3, 0.5, 0.7]`：会分别计算这三个阈值下的召回率
- 可以根据需要修改，例如：`[0.3, 0.5, 0.7, 0.9]`（增加更严格的标准）

---

### 3. EVAL_METRIC 参数

```yaml
EVAL_METRIC: kitti  # 使用 kitti 格式输出评估指标
```

**含义**：
- 指定评估方法的类型
- `kitti`：使用KITTI数据集的评估标准（计算AP、mAP等指标）

**可选值**：
- `kitti`：KITTI评估标准（**推荐用于3D检测**）
  - 计算BBox AP、BEV AP、3D AP
  - 支持多难度级别（Easy、Moderate、Hard）
  - 计算不同IoU阈值下的性能

- 其他数据集标准（需要实现相应的evaluation方法）：
  - `waymo`
  - `nuscenes`
  - `custom`

**为什么使用KITTI标准**：
- KITTI是3D目标检测的经典评估标准
- 即使你的数据集不是KITTI，也可以使用其评估方法
- 提供详细的AP（Average Precision）指标

---

## 📈 完整评估输出（修复后会看到）

修复`evaluation`方法后，你会看到完整的评估结果，类似：

```
Box AP@0.70, 0.70, 0.70:
bbox AP: 0.8745, 0.7823, 0.6921
bev  AP: 0.8512, 0.7634, 0.6732
3d   AP: 0.7823, 0.6945, 0.5834

Diver AP@0.70, 0.70, 0.70:
bbox AP: 0.7934, 0.6812, 0.5923
bev  AP: 0.7654, 0.6543, 0.5712
3d   AP: 0.6923, 0.5834, 0.4921
```

### 指标含义

#### 1. AP (Average Precision)

**定义**：在不同召回率下精确率的平均值

**三个维度的AP**：

1. **bbox AP**（2D边界框AP）
   - 评估2D投影框的检测性能
   - 对于3D点云不太重要

2. **bev AP**（鸟瞰图AP，Bird's Eye View）
   - 评估XY平面上的检测性能
   - 忽略高度信息
   - 对于声纳数据比较重要（水平定位）

3. **3d AP**（3D边界框AP）
   - **最重要的指标**
   - 评估完整3D空间的检测性能
   - 同时考虑X、Y、Z、长宽高、旋转角

#### 2. 三列数字含义

```
3d AP: 0.7823, 0.6945, 0.5834
       ↓       ↓       ↓
      Easy   Moderate  Hard
```

**难度级别**（对于KITTI标准）：
- **Easy**: 大目标、无遮挡、清晰可见
- **Moderate**: 中等大小、部分遮挡
- **Hard**: 小目标、严重遮挡

**对于声纳数据**：
- 可能根据目标距离、点云密度、噪声程度划分难度
- 远距离、稀疏点云 → Hard
- 近距离、密集点云 → Easy

---

## 🎯 如何解读你的评估结果

### 当前输出分析

```
recall_rcnn_0.3: 0.835660  # 83.57% 召回率（宽松标准）
recall_rcnn_0.5: 0.652352  # 65.24% 召回率（中等标准）
recall_rcnn_0.7: 0.387253  # 38.73% 召回率（严格标准）
Average predicted number of objects: 1.763  # 平均每帧预测1.763个目标
```

### 性能评估

#### ✅ 优点
- **高召回率**（宽松标准）：能检测到大部分目标
- **平均预测数量合理**：1.763个/帧（数据集中每帧确实有1-2个目标）

#### ⚠️ 需要改进
- **定位精度不足**：
  - IoU 0.5→0.7时召回率从65%降到39%
  - 说明预测框位置/大小不够精确
  
- **可能原因**：
  1. 训练epoch数不足（24/80）
  2. 声纳数据稀疏导致定位困难
  3. 体素化参数需要调整
  4. 需要更多训练数据

---

## 🔧 如何提升评估指标

### 1. 提高召回率（Recall）

**策略**：
- 降低`SCORE_THRESH`（更宽松的检测阈值）
  ```yaml
  SCORE_THRESH: 0.1  # 降低到 0.05
  ```
- 增加`MAX_NUMBER_OF_VOXELS`（处理更多点云）
- 使用数据增强

### 2. 提高定位精度（IoU 0.7下的Recall）

**策略**：
- **减小体素尺寸**（更精细的网格）
  ```yaml
  VOXEL_SIZE: [0.25, 0.4, 0.35]  # 改为 [0.2, 0.3, 0.3]
  ```
- **增加训练epoch**：24 → 80
- **调整损失权重**：
  ```yaml
  LOSS_CONFIG:
      LOC_WEIGHT: 2.0  # 提高定位损失权重
  ```

### 3. 提高整体mAP

**策略**：
- 继续训练更多epoch
- 使用更大的batch_size（梯度累积）
- 调整学习率策略
- 数据增强（已禁用）

---

## 📊 与其他模型对比

### KITTI数据集基准（Car类）

| 模型 | 3D AP (Moderate) | 速度 |
|------|-----------------|------|
| PointPillars | ~68% | 快 |
| SECOND | ~78% | 中 |
| **Focal SST** | ~82% | 慢 |
| PV-RCNN | ~83% | 很慢 |

### 你的声纳数据表现

- **当前**：recall@0.5 = 65.24%
- **目标**：通过完整训练达到 70-80%

---

## 🛠️ 修复后的完整评估流程

### 1. 训练完成后自动评估

```python
# train.py 会自动调用
repeat_eval_ckpt(
    model=model,
    test_loader=test_loader,
    args=args,
    eval_output_dir=eval_output_dir,
    logger=logger,
    ckpt_dir=ckpt_dir,
    dist_test=dist_test
)
```

### 2. 评估调用链

```
train.py
  └─> repeat_eval_ckpt() [test.py]
      └─> eval_one_epoch() [eval_utils/eval_utils.py]
          └─> dataset.evaluation()  [sonar_dataset.py]
              └─> kitti_eval.get_official_eval_result()
```

### 3. 输出文件

评估结果会保存在：
```
output/sonar_models/focal_sst/default/
├── eval/
│   └── epoch_24/
│       ├── result.pkl          # 检测结果
│       └── final_result/
│           └── data/           # 详细评估数据
```

---

## 💡 常见问题

### Q1: 为什么recall_roi全是0？

**A**: CenterPoint是**anchor-free单阶段检测器**，没有单独的RoI生成阶段，所以`recall_roi`为0是正常的。只需关注`recall_rcnn`即可。

### Q2: 为什么有些类的AP是0？

**A**: 可能原因：
- 该类在验证集中样本太少
- 模型没有学习到该类特征
- 类别名称不匹配

### Q3: 如何只评估特定类别？

**A**: 修改配置：
```yaml
CLASS_NAMES: ['Box']  # 只评估Box类
# 或
CLASS_NAMES: ['Box', 'Diver']  # 评估两个类
```

### Q4: 评估太慢怎么办？

**A**: 
```yaml
# 只评估部分样本（测试用）
# 修改 ImageSets/val.txt，减少样本数量
```

---

## 📝 总结

### 关键指标优先级（对于3D检测）

1. **3D AP @ IoU 0.5**（最重要）
2. **recall_rcnn @ 0.5**（召回率）
3. **BEV AP**（水平定位）
4. **precision @ 0.5**（精确率）

### 你的模型现状

- ✅ 召回率良好（宽松标准83%）
- ⚠️ 定位精度待提升（严格标准39%）
- ⏳ 训练尚未完成（24/80 epoch）

### 下一步建议

1. ✅ **继续训练**：完成剩余56个epoch
2. 📊 **观察趋势**：查看TensorBoard中AP曲线
3. 🔧 **调整参数**：如果80 epoch后仍不理想，尝试：
   - 减小体素尺寸
   - 调整损失权重
   - 增加训练数据

---

## 附录：完整配置示例

```yaml
# focal_sst.yaml
CLASS_NAMES: ['Box', 'Diver']

POST_PROCESSING:
    RECALL_THRESH_LIST: [0.3, 0.5, 0.7]  # 召回率评估阈值
    SCORE_THRESH: 0.1                    # 检测置信度阈值
    OUTPUT_RAW_SCORE: False
    EVAL_METRIC: kitti                   # 使用KITTI评估标准

    NMS_CONFIG:
        MULTI_CLASSES_NMS: False
        NMS_TYPE: nms_gpu
        NMS_THRESH: 0.1
        NMS_PRE_MAXSIZE: 4096
        NMS_POST_MAXSIZE: 500
```

现在evaluation方法已修复，重新运行训练时评估会正常输出完整的AP指标！
