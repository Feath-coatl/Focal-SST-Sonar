# Focal SST 模型技术架构设计方案

## 1. 模型总体架构(Overall Architecture)
### **概述**
**Focal SST**是一种专为水下声呐3D感知设计的检测架构。它融合了TANet的抗噪特征提取能力、Focal Sparse Conv 的动态结构生成能力以及DSVT的高效全局上下文建模能力。该模型旨在解决声呐图像中“目标特征被散斑噪声淹没”以及“稀疏点云导致特征断裂”的核心痛点，实现从特征去噪到结构补全，再到全局长程依赖建模的端到端优化。

**架构图解描述**
数据流向设计如下：
1. **Input**:输入原始声呐点云（包含$x,y,z,intensity$）。
2. **Dynamic Voxelization**:进行动态体素化，生成初始Voxel Features。
3. **VFE(Feature Encoding)**:进入SDAVFE(Stacked Double Attention VFE)模块，利用双重注意力机制在体素内部进行特征增强和噪声抑制。
4. **Stem/Encoder**:进入Focal Sparse Encoder，通过预测体素重要性，动态地对稀疏目标进行致密化（Dilation），填补声呐盲区导致的结构缺失。
5. **Backbone**:进入DSVT Transformer主干，利用动态稀疏窗口和旋转集合划分策略，高效处理长程上下文信息。
6. **Neck/Head**:输出至BEV Grid，连接检测头（如CenterHead）输出3D边界框。

## 2. 核心模块详细设计(Module Design & Rationale)
### 2.1. 特征编码层：SDAVFE(Stacked Double Attention VFE)
- 来源:基于TANet(Triple Attention Network)的改进。
- 设计逻辑:
  - TANet原文提出了Channel-wise, Point-wise和Voxel-wise三重注意力。
  - **改进设计**:在 Focal SST中，我们将其简化为Stacked Double Attention(SDAVFE)，即保留Channel-wise Attention和Point-wise Attention。
  - **理由**:Voxel-wise Attention(体素级全局加权)的功能与后续的Focal Sparse Conv(体素重要性预测)功能重叠。为了降低计算冗余，我们在VFE阶段专注于体素内部的细粒度特征清洗，将体素之间的重要性判断留给Focal层。
- 声呐适配性:
  - **输入扩展**:按照规划，输入向量扩展为10D(另外6个维度为该点距离体素中心和体素重心的距离)，未来也可能额外增加掠射角(Grazing Angle, $\theta_g$)和距离(Range, $R$)特征进行实验(目前暂无该计划)。
### 2.2. 稀疏几何特征提取：Focal Sparse Encoder
- 来源:源自Focal Sparse Convolution。
- 设计逻辑:
  - 传统稀疏卷积会无差别地膨胀所有特征，导致计算量激增且引入背景噪声；Submanifold Sparse Conv虽然高效但无法通过“空洞”传播信息。
  - **Focal 机制**:引入一个轻量级的子分支预测“体素重要性”($I_p$)。只有被预测为“重要”的前景体素（$I_p > \tau$）才会进行卷积膨胀（Dilation），生成新的结构特征；背景体素则保持稀疏或被抑制。
- 优势与声呐适配:
  - **解决“空洞盲区”**:声呐点云常因遮挡或波束间隔导致目标特征在空间上离散（例如自行车的前后轮中间有空隙）。Focal Conv可以主动在这些“空隙”处生成特征，起到物理上的“桥接”作用。
  - **训练策略**:在训练时，利用Ground Truth生成掩码。
### 2.3. Transformer主干：DSVT-based Backbone
- 来源:源自DSVT(Dynamic Sparse Voxel Transformer)。
- 设计逻辑:
    - 痛点:原版SST模型将体素划分为局部窗口，为了并行计算，必须对稀疏窗口进行Padding（填充虚假Token），这在极度稀疏的声呐数据中导致了大量的算力浪费。
    - DSVT解决方案:
      1. **动态集合划分(Dynamic Set Partition)**:不再进行填充，而是将窗口内的有效体素排序并划分为大小相等的子集（Sets），实现完全并行的注意力计算。
      2. **旋转集合(Rotated Sets)**:在连续的层之间，交替使用不同的划分轴（例如先X轴主序，再Y轴主序）。这使得信息可以在不同的集合之间流动，建立了跨窗口的联系。
- 声呐适配性:
    - **Range-Azimuth划分**:用户规划中创新地提出，不仅可以使用DSVT已有的笛卡尔坐标系(X-Y)旋转，还可以尝试引入距离-方位(Range-Azimuth)视图的旋转划分。这允许Transformer建模传感器原生的关系（如径向对齐的多径效应）和世界坐标系的关系。
## 3. 综合优化与创新(Synthesis&Innovation)
### 3.1 协同效应(Synergy)
Focal SST 并非模块的简单堆砌，而是针对声呐成像链路的针对性重构：
1. **去噪(SDAVFE)**:先在最前端利用注意力机制压制声呐的散斑噪声，防止噪声在后续层被放大。
2. **结构恢复(Focal Conv)**:在特征纯净后，利用Focal机制动态填补声呐成像的离散空洞，为Transformer提供连续的语义特征。
3. **高效建模(DSVT)**:最后利用DSVT处理不规则分布的稀疏特征，无需Padding，极大提升了推理速度，适合水下机器人的算力受限环境。
### 3.2 解决核心痛点
- **针对噪声**:结合TANet的点级注意力和Focal Conv的重要性筛选，实现了“特征内去噪”和“空间上去噪”的双重过滤。
- **针对稀疏性**:Focal Sparse Conv的动态膨胀机制，使得模型能够自适应地致密化稀疏的小目标，避免漏检。
- **针对效率**:DSVT彻底消除了处理稀疏声呐数据时的冗余计算，使得模型在实时计算部署上成为可能。
## 4. 潜在的改进方向
1. **各向异性扩张**:在Focal Sparse Conv中，标准的Dilation是各向同性的（3x3x3立方体）。针对声呐点云数据特性，可以设计各向异性的扩张核，限制沿Range方向的扩张，鼓励沿Azimuth方向的扩张，提高新生成特征的可信度。
2. **IoU-Rectification Loss**:引入IoU预测分支来校正置信度分数。由于声呐缺乏纹理，边界框回归往往不准。通过预测3D IoU并将其作为置信度的加权项，可以显著提升mAP指标。