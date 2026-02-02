# 浮点异常诊断报告

## 1. 问题现象
- 错误类型: `Floating point exception (core dumped)`
- 错误时机: 训练过程中加载数据时
- 问题样本: a_0180077.txt, d_0170751.txt, c_0140342.txt

## 2. 已验证的正常部分

### 2.1 数据文件完全正常
- ✅ 所有3个问题文件都可以正常加载
- ✅ intensity值范围正常: [4e8 ~ 6.27e10]
- ✅ 没有NaN或Inf值
- ✅ 点数正常: ~40,000点/帧

### 2.2 get_lidar()方法完全正常  
- ✅ np.loadtxt 成功
- ✅ np.clip 成功
- ✅ np.log10 成功
- ✅ 除法运算成功
- ✅ 坐标对齐成功
- ✅ 最终points无NaN/Inf

## 3. 问题定位

**核心发现**: 浮点异常不是由Python代码触发的，而是**硬件级别的SIGFPE信号**。

这种异常通常由以下情况触发：
1. **C/C++扩展中的整数除以0**
2. **C/C++扩展中的整数溢出**  
3. **spconv/cumm等C++库中的浮点异常**

### 3.1 问题最可能出现在
根据代码流程分析，浮点异常最可能发生在：

**prepare_data() 的三个步骤中**:
1. `data_augmentor.forward()` - 数据增强（旋转、缩放等）
2. `point_feature_encoder.forward()` - 特征编码
3. **`data_processor.forward()` - voxelization** ← **最可疑**

### 3.2 为什么怀疑voxelization?
1. spconv的voxelization是用C++/CUDA实现的
2. C++代码中可能有除法操作，如果除数为0会触发SIGFPE
3. voxelization过程会计算坐标->voxel索引的映射，涉及除法
4. 如果点云范围超出预期，可能导致计算异常

## 4. 下一步建议

### 方案A: 定位具体步骤
在prepare_data中添加分步调试，看哪一步触发异常：
```python
def prepare_data(self, data_dict):
    print(f"[DEBUG] 开始prepare_data, frame={data_dict.get('frame_id')}")
    
    if self.training:
        print("[DEBUG] Step 1: data_augmentor")
        data_dict = self.data_augmentor.forward(data_dict)
        
    if data_dict.get('points', None) is not None:
        print("[DEBUG] Step 2: point_feature_encoder")
        data_dict = self.point_feature_encoder.forward(data_dict)
    
    print("[DEBUG] Step 3: data_processor (voxelization)")
    data_dict = self.data_processor.forward(data_dict)
    
    return data_dict
```

### 方案B: 检查点云范围
检查问题样本的点云是否在配置的POINT_CLOUD_RANGE内：
```yaml
POINT_CLOUD_RANGE: [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
```

如果点云超出范围，voxelization时可能会计算出异常的voxel索引。

### 方案C: 使用gdb调试
```bash
gdb --args python tools/train.py --cfg_file tools/cfgs/sonar_models/focal_sst.yaml
```
当crash时查看具体的C++堆栈。

### 方案D: 检查spconv版本
不同版本的spconv可能有不同的bug，考虑：
1. 升级/降级spconv版本
2. 查看spconv的issue列表

## 5. 临时workaround (不推荐)

如果需要先让训练跑起来，可以：
1. 在__getitem__中catch浮点异常（虽然catch不到SIGFPE，但可以try多次）
2. 或者暂时跳过问题样本（在dataset中过滤掉）

**但这治标不治本，建议还是定位根因。**
