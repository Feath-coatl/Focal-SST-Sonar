# 二进制格式转换使用指南

## 🎯 为什么转换为二进制格式？

### 性能提升
- **读取速度**: 5-10倍提升（`np.fromfile()` vs `np.loadtxt()`）
- **磁盘空间**: 节省75%（19.87 GB → ~5 GB）
- **内存映射**: 直接加载，无需字符串解析

### 数据集现状
- **文件数量**: 12,220个
- **当前大小**: 19.87 GB（文本格式）
- **预计转换后**: ~5 GB（二进制格式）
- **平均文件**: 1.7 MB → 0.4 MB

## 📋 使用步骤

### 步骤1: 转换数据集

```bash
# 进入项目目录
cd d:\Desktop\thesis\Modelproject\MainFocal\focalSST-master

# 转换全部数据集（约需5-10分钟）
python convert_txt_to_binary.py --data_path data/sonar/points

# 测试模式（仅转换前10个文件）
python convert_txt_to_binary.py --data_path data/sonar/points --test_mode
```

**输出目录**：`data/sonar/points_binary/`

### 步骤2: 修改配置文件

编辑 `tools/cfgs/dataset_configs/sonar_dataset.yaml`：

```yaml
# 启用二进制格式
USE_BINARY_FORMAT: True
```

### 步骤3: 正常训练

```bash
cd tools
python train.py --cfg_file cfgs/sonar_models/focal_sst.yaml --batch_size 2 --workers 4 --epochs 80
```

## 📊 性能对比

### 磁盘空间

| 格式 | 大小 | 节省 |
|------|------|------|
| 文本(.txt) | 19.87 GB | - |
| 二进制(.bin) | ~5 GB | **75%** |

### 读取速度（单文件）

| 格式 | 平均耗时 | 提升 |
|------|---------|------|
| 文本(.txt) | ~30-50 ms | - |
| 二进制(.bin) | ~3-5 ms | **5-10x** |

### 训练影响

**当前瓶颈分析**：
- 模型计算: 88%
- 数据加载: 12%

**预计加速效果**：
- 数据加载部分: 5-10倍加速
- 整体训练速度: 提升约5-10%（因为数据加载只占12%）
- 第一个epoch: 明显更快（无需文本解析）

## 🔧 技术细节

### 文件格式

**文本格式(.txt)**：
```
26.892200 4.011570 1.049250 621316992.0 1.0
26.856701 4.193640 1.231300 806521984.0 1.0
...
```
- 每个float: ~18字节（包括空格、小数点、科学计数法）
- 需要字符串解析

**二进制格式(.bin)**：
```
[binary data: float32 array, shape=(N, 5)]
```
- 每个float32: 4字节
- 直接内存映射，无需解析

### 代码实现

**保存（转换脚本）**：
```python
points = np.loadtxt('file.txt', dtype=np.float32)  # [N, 5]
points.tofile('file.bin')  # 直接写入二进制
```

**加载（sonar_dataset.py）**：
```python
if self.use_binary_format:
    points = np.fromfile('file.bin', dtype=np.float32).reshape(-1, 5)
else:
    points = np.loadtxt('file.txt', dtype=np.float32)
```

## ⚠️ 注意事项

### 1. 磁盘空间
- 转换过程中需要 **额外5GB空间**（新旧文件共存）
- 完成后可选择性删除.txt文件（建议保留备份）

### 2. 数据增强文件
如果你的增强数据也是.txt格式，也需要转换：
```bash
# 转换增强数据
python convert_txt_to_binary.py --data_path data/sonar/points_augmented
```

### 3. 可视化工具
如果需要用可视化工具查看点云，需要：
- **方案A**: 保留部分.txt文件用于可视化
- **方案B**: 修改可视化脚本支持.bin格式（小改动）

### 4. 回退方案
如果遇到问题，可随时切换回文本格式：
```yaml
USE_BINARY_FORMAT: False  # 恢复使用.txt文件
```

## 🎯 推荐使用场景

### ✅ 适合使用二进制格式
- ✅ 数据集已经固定，不再修改
- ✅ 磁盘空间有限（节省15GB）
- ✅ 追求最佳训练速度
- ✅ 不需要频繁手动查看点云数据

### ⚠️ 暂时使用文本格式
- ⚠️ 数据集仍在调整中
- ⚠️ 需要频繁查看原始数据
- ⚠️ 磁盘空间充足

## 🔍 转换后验证

### 验证数据完整性

```python
# 测试脚本
import numpy as np

# 读取文本文件
txt_data = np.loadtxt('data/sonar/points/0130001.txt', dtype=np.float32)

# 读取二进制文件
bin_data = np.fromfile('data/sonar/points_binary/0130001.bin', dtype=np.float32).reshape(-1, 5)

# 验证数据一致性
print(f"形状匹配: {txt_data.shape == bin_data.shape}")
print(f"数值匹配: {np.allclose(txt_data, bin_data)}")
print(f"最大差异: {np.max(np.abs(txt_data - bin_data))}")
```

预期输出：
```
形状匹配: True
数值匹配: True
最大差异: 0.0
```

### 速度测试

转换脚本会自动进行速度测试并显示结果：
```
文件: 0130001.txt
文本读取(.txt):     45.23 ms
二进制读取(.bin):   4.87 ms
速度提升:           9.3x
```

## 📈 预期效果

### 对于你的数据集（12,220个文件）

#### 第一个epoch（无缓存）
- **当前**: ~3.5小时（包含文本解析开销）
- **转换后**: ~3.2小时（减少文本解析时间）
- **提升**: 约10%

#### 使用内存缓存（CACHE_ALL_DATA=True）
- **影响**: 第一个epoch加速10%，后续epoch无明显差异（已缓存）

#### 综合优势
1. **磁盘节省15GB**（主要优势）
2. **第一个epoch更快**（次要优势）
3. **降低CPU负载**（减少解析开销）
4. **更好的I/O性能**（对NAS/网络存储更友好）

## 🚀 快速测试

想先测试效果？使用测试模式：

```bash
# 1. 转换前10个文件
python convert_txt_to_binary.py --data_path data/sonar/points --test_mode

# 2. 创建测试数据集配置
cp tools/cfgs/dataset_configs/sonar_dataset.yaml tools/cfgs/dataset_configs/sonar_dataset_test.yaml

# 3. 在测试配置中添加
# INFO_PATH: {train: [sonar_infos_train.pkl], test: [sonar_infos_val.pkl]}
# 并手动编辑.pkl文件只保留前10个样本

# 4. 对比测试
# 文本格式
USE_BINARY_FORMAT: False
# vs
# 二进制格式  
USE_BINARY_FORMAT: True
```

## 💡 总结

**推荐操作**：
1. ✅ **立即转换**：节省15GB磁盘空间
2. ✅ **测试验证**：确保数据一致性
3. ✅ **正式使用**：享受5-10倍读取速度

**风险评估**：
- 🟢 技术风险：极低（成熟方案）
- 🟢 数据风险：无（可验证一致性）
- 🟢 性能风险：无（只会更快）
- 🟡 存储风险：需额外5GB临时空间

**时间成本**：
- 转换时间：5-10分钟
- 验证时间：2-3分钟
- 配置修改：1分钟

**收益**：
- 磁盘节省：15GB（永久）
- 速度提升：10%（每次训练）
- CPU降低：减少文本解析负担
