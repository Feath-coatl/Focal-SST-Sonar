# 训练断点续训与实验管理指南

## 问题1：为什么有时继续训练，有时重新开始？

### 原因分析

训练代码会自动查找checkpoint并继续训练（train.py#L157-166）：

```python
# 如果没有指定--ckpt参数，会自动查找最新的checkpoint
ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
if len(ckpt_list) > 0:
    ckpt_list.sort(key=os.path.getmtime)  # 按修改时间排序
    # 自动加载最新的checkpoint
```

### 你遇到的情况

- ✅ **继续训练**：checkpoint文件存在 → 自动加载
- ❌ **重新开始**：可能的原因：
  1. 修改了`--extra_tag`参数（输出目录改变）
  2. checkpoint文件被删除或移动
  3. checkpoint加载失败（配置不匹配）

### 输出目录结构

```
output/sonar_models/focal_sst/default/
├── ckpt/
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   └── ...
├── tensorboard/
└── log_train_*.txt
```

路径由以下部分组成：
- `sonar_models` ← cfg.EXP_GROUP_PATH（配置文件路径）
- `focal_sst` ← cfg.TAG（配置文件名）
- `default` ← args.extra_tag（你指定的标签）

---

## 问题2：如何保存和区分多次训练？

### 使用`--extra_tag`参数来区分不同的实验

上一次训练结果**不会消失**，所有实验结果都保存在各自的目录中。

---

## 完整解决方案

### 情况1：修改配置后做新实验

```bash
# 第一次训练（基线）
python train.py \
    --cfg_file cfgs/sonar_models/focal_sst.yaml \
    --batch_size 2 \
    --workers 4 \
    --epochs 80 \
    --extra_tag baseline \
    --fp16

# 修改focal_sst.yaml后的第二次训练
python train.py \
    --cfg_file cfgs/sonar_models/focal_sst.yaml \
    --batch_size 2 \
    --workers 4 \
    --epochs 80 \
    --extra_tag grad_accum_x4 \
    --fp16
```

**结果**：
```
output/sonar_models/focal_sst/
├── baseline/              ← 第一次训练的结果
│   ├── ckpt/
│   ├── tensorboard/
│   └── log_train_*.txt
└── grad_accum_x4/         ← 第二次训练的结果
    ├── ckpt/
    ├── tensorboard/
    └── log_train_*.txt
```

### 情况2：中断后继续训练

**自动继续**（推荐）：
```bash
# 重新运行相同的命令，会自动加载最新checkpoint
python train.py \
    --cfg_file cfgs/sonar_models/focal_sst.yaml \
    --batch_size 2 \
    --workers 4 \
    --epochs 80 \
    --extra_tag baseline \
    --fp16
```

**手动指定checkpoint**：
```bash
python train.py \
    --cfg_file cfgs/sonar_models/focal_sst.yaml \
    --batch_size 2 \
    --workers 4 \
    --epochs 80 \
    --extra_tag baseline \
    --ckpt output/sonar_models/focal_sst/baseline/ckpt/checkpoint_epoch_10.pth \
    --fp16
```

### 情况3：从头重新训练

**方法1：删除checkpoint目录**
```powershell
Remove-Item output/sonar_models/focal_sst/baseline/ckpt/* -Recurse
```

**方法2：使用新的extra_tag**
```bash
python train.py \
    --extra_tag baseline_v2
```

### 情况4：基于已有模型fine-tune

```bash
python train.py \
    --cfg_file cfgs/sonar_models/focal_sst.yaml \
    --pretrained_model output/sonar_models/focal_sst/baseline/ckpt/checkpoint_epoch_80.pth \
    --extra_tag finetune_exp \
    --epochs 100 \
    --fp16
```

---

## 关于配置修改的影响

### ✅ 可以安全修改（不影响继续训练）

以下参数修改后，仍可继续训练：

- `BATCH_SIZE_PER_GPU`（通过命令行`--batch_size`覆盖）
- `NUM_EPOCHS`（通过命令行`--epochs`覆盖）
- `GRAD_ACCUMULATION_STEPS`
- 学习率相关参数（`LR`, `WARMUP_EPOCH`, `GRAD_NORM_CLIP`等）
- `workers`数量
- 数据增强参数

### ⚠️ 修改后无法继续训练

以下参数修改后会导致checkpoint加载失败，**必须使用新的`--extra_tag`从头训练**：

#### 模型结构参数
- `d_model`（特征维度）
- `nhead`（注意力头数）
- `set_info`（DSVT集合大小）
- `window_shape`（窗口大小）
- `conv_out_channel`（输出通道数）

**原因**：模型权重维度不匹配

#### 体素化参数
- `VOXEL_SIZE`（体素大小）
- `MAX_NUMBER_OF_VOXELS`（体素数量）
- `MAX_POINTS_PER_VOXEL`（每个体素最大点数）

**原因**：模型输入维度变化，`sparse_shape`需要相应调整

#### 点云范围
- `POINT_CLOUD_RANGE`

**原因**：影响体素网格尺寸

---

## 实用建议

### 1. 使用有意义的extra_tag

```bash
# ❌ 不好的命名
--extra_tag default
--extra_tag test1
--extra_tag new

# ✅ 好的命名（一眼看出实验内容）
--extra_tag baseline_bs2
--extra_tag grad_accum_x4_lr0.0005
--extra_tag small_model_d96
--extra_tag voxel_0.15_workers8
--extra_tag exp1_baseline_20260205
```

### 2. 创建实验记录文件

在项目根目录创建 `experiments_log.txt`：
```
日期       | 标签                    | 配置说明                          | 结果
-----------|------------------------|----------------------------------|------
2026-02-05 | baseline_bs2           | batch_size=2, workers=4, 基线    | mAP: 0.xx
2026-02-05 | grad_accum_x4          | 梯度累积x4, LR=0.0005           | mAP: 0.xx
2026-02-06 | small_model_d96        | d_model=96, nhead=6, 轻量化     | mAP: 0.xx
2026-02-06 | voxel_0.15             | VOXEL_SIZE=[0.15,0.25,0.25]     | mAP: 0.xx
```

### 3. 检查checkpoint是否存在

```powershell
# 查看某个实验的所有checkpoint
Get-ChildItem output/sonar_models/focal_sst/baseline/ckpt/

# 查看最新的checkpoint（按修改时间排序）
Get-ChildItem output/sonar_models/focal_sst/baseline/ckpt/*.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### 4. 对比不同实验的结果

使用TensorBoard同时查看多个实验：
```bash
tensorboard --logdir=output/sonar_models/focal_sst/ --port=6006
```

会同时显示所有实验的曲线，方便对比。

### 5. 备份重要的checkpoint

```powershell
# 备份最佳模型
Copy-Item output/sonar_models/focal_sst/baseline/ckpt/checkpoint_epoch_80.pth `
          output/sonar_models/focal_sst/baseline/ckpt/best_model_backup.pth

# 备份整个实验目录
Copy-Item -Recurse output/sonar_models/focal_sst/baseline `
          output/sonar_models/focal_sst/baseline_backup_20260205
```

---

## 常见场景示例

### 场景1：调参实验（保留基线）

```bash
# 基线实验
python train.py --extra_tag baseline --epochs 80 --fp16

# 实验1：尝试更大的学习率
python train.py --extra_tag exp1_lr0.002 --epochs 80 --fp16
# 修改focal_sst.yaml: LR: 0.002

# 实验2：尝试梯度累积
python train.py --extra_tag exp2_accum_x4 --epochs 80 --fp16
# 修改focal_sst.yaml: GRAD_ACCUMULATION_STEPS: 4

# 基线仍然保留在 baseline/ 目录下
```

### 场景2：模型消融实验

```bash
# 完整模型
python train.py --extra_tag full_model --epochs 80 --fp16

# 去掉Focal机制（需要修改模型配置）
python train.py --extra_tag ablation_no_focal --epochs 80 --fp16

# 使用更小的模型
python train.py --extra_tag ablation_small_d96 --epochs 80 --fp16
```

### 场景3：训练中断恢复

```bash
# 原始训练命令
python train.py --extra_tag baseline --epochs 80 --fp16

# 训练在epoch 30时中断...

# 恢复训练（使用完全相同的命令）
python train.py --extra_tag baseline --epochs 80 --fp16
# 会自动从checkpoint_epoch_30.pth继续
```

### 场景4：从最佳模型继续训练更多epoch

```bash
# 原始训练了80个epoch
python train.py --extra_tag baseline --epochs 80 --fp16

# 继续训练到120个epoch
python train.py --extra_tag baseline --epochs 120 --fp16
# 会从epoch 80继续训练到120
```

---

## 故障排查

### 问题1：训练总是从头开始

**检查**：
```powershell
# 1. 确认checkpoint目录是否存在
Test-Path output/sonar_models/focal_sst/YOUR_TAG/ckpt/

# 2. 查看是否有checkpoint文件
Get-ChildItem output/sonar_models/focal_sst/YOUR_TAG/ckpt/*.pth

# 3. 确认extra_tag是否正确
# 检查训练日志中的路径
```

**原因**：
- `--extra_tag`不匹配
- checkpoint被意外删除
- 路径不正确

### 问题2：加载checkpoint时报错

**错误信息示例**：
```
RuntimeError: Error(s) in loading state_dict...
size mismatch for xxx: copying a param with shape...
```

**原因**：模型配置已修改，权重维度不匹配

**解决**：使用新的`--extra_tag`从头训练

### 问题3：想强制从头训练

**方法1**：删除checkpoint
```powershell
Remove-Item output/sonar_models/focal_sst/YOUR_TAG/ckpt/*.pth
```

**方法2**：使用新的extra_tag
```bash
python train.py --extra_tag YOUR_TAG_v2 --epochs 80 --fp16
```

---

## 总结

### 核心原则

1. **不同实验用不同的`--extra_tag`** → 结果互不影响
2. **相同的`--extra_tag`** → 自动断点续训
3. **修改模型结构** → 必须用新的`--extra_tag`
4. **只修改超参数** → 可以继续训练（但不推荐）

### 推荐工作流

```bash
# 1. 基线实验
python train.py --extra_tag baseline_20260205 --epochs 80 --fp16

# 2. 改进实验1
# 修改配置 → 使用新tag
python train.py --extra_tag improved_v1_20260206 --epochs 80 --fp16

# 3. 改进实验2
# 再次修改 → 再用新tag
python train.py --extra_tag improved_v2_20260207 --epochs 80 --fp16

# 4. 对比结果
tensorboard --logdir=output/sonar_models/focal_sst/
```

### 最佳实践

- ✅ 每次实验使用**日期+描述**的tag
- ✅ 记录实验配置到日志文件
- ✅ 重要模型及时备份
- ✅ 使用TensorBoard对比实验
- ✅ 定期清理不需要的checkpoint（保留最佳模型）

### 避免的错误

- ❌ 所有实验都用`default`作为tag
- ❌ 修改配置后不换tag（导致加载失败）
- ❌ 不记录实验配置（事后忘记参数）
- ❌ 删除checkpoint后期望继续训练
