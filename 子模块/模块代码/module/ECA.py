# import torch
# from torch import nn

# class ECA_layer(nn.Module):
#     """构建一个 ECA 模块。

#     参数:
#         channel: 输入特征图的通道数
#         k_size: 自适应选择的一维卷积核大小
#     """
#     def __init__(self, channel, k_size=3):
#         super(ECA_layer, self).__init__()
        
#         # 全局平均池化层，用于将每个通道的空间信息压缩成一个值
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # 一维卷积层，用于捕捉通道之间的交互信息
#         # 1. 输入通道数为1，因为经过全局平均池化后，每个特征图都变成了1x1
#         # 2. 输出通道数为1，因为我们不想改变通道数量，只是调整权重
#         # 3. kernel_size=k_size，指定卷积核的大小
#         # 4. padding=(k_size - 1) // 2，用于保持卷积后的张量长度与输入一致
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        
#         # Sigmoid激活函数，将输出的范围限制在(0, 1)之间
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         """
#         前向传播函数，定义数据流经过该模块的处理步骤。

#         参数:
#         x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

#         返回:
#         Tensor: 经过ECA模块处理后的输出张量。
#         """
        
#         # 使用全局平均池化将每个通道的空间维度 (H, W) 压缩到 1x1
#         # 输出张量的形状将变为 [batch_size, channels, 1, 1]
#         y = self.avg_pool(x)
        
#         # 去掉最后一个维度，并交换第二个和第三个维度
#         # y.squeeze(-1) 的形状是 [batch_size, channels, 1]
#         # y.transpose(-1, -2) 交换后的形状是 [batch_size, 1, channels]
#         y = y.squeeze(-1).transpose(-1, -2)
        
#         # 通过一维卷积处理，卷积核大小是 k_size
#         # 形状保持 [batch_size, 1, channels]，内容经过一维卷积核处理
#         y = self.conv(y)
        
#         # 再次交换维度，恢复原始的通道顺序
#         # y.transpose(-1, -2) 将形状从 [batch_size, 1, channels] 变为 [batch_size, channels, 1]
#         y = y.transpose(-1, -2)
        
#         # 恢复被去掉的维度，将形状从 [batch_size, channels, 1] 变为 [batch_size, channels, 1, 1]
#         y = y.unsqueeze(-1)
        
#         # 使用 Sigmoid 激活函数将输出限制在 (0, 1) 之间
#         y = self.sigmoid(y)
        
#         # 将输入张量 x 与处理后的权重 y 相乘，进行通道加权
#         # expand_as 确保 y 的形状与 x 匹配，以便逐元素相乘
#         return x * y.expand_as(x)

# # 示例用法
# if __name__ == "__main__":
#     # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
#     x = torch.randn(4, 64, 32, 32)
#     # 创建一个 ECA 模块实例，通道数为 64
#     eca = ECA_layer(channel=64)
#     # 通过 ECA 模块调整输入特征
#     y = eca(x)
#     # 打印输出张量的形状，应该与输入相同
#     print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])


# import torch
# from torch import nn

# class DSCA_layer(nn.Module):
#     def __init__(self, channel, k_size=3):
#         super(DSCA_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出形状为[B, C, 1, 1]

#         # 替换为深度可分离卷积
#         self.depthwise_conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, groups=1, bias=False)  
#         # 深度可分离卷积：分开处理每个通道的卷积操作，`groups=1`表示每个通道单独卷积
#         self.pointwise_conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)  # 1x1卷积，用于进一步融合通道特征

#         self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成通道注意力权重

#     def forward(self, x):
#         y = self.avg_pool(x)  # 对输入进行全局平均池化，得到形状为[B, C, 1, 1]的张量
#         y = y.squeeze(-1).transpose(-1, -2)  # 去掉最后一个维度并交换最后两个维度，得到形状为[B, 1, C]

#         # 深度可分离卷积
#         y = self.depthwise_conv(y)  # 对每个通道的特征进行深度可分离卷积
#         y = self.pointwise_conv(y)  # 进一步通过1x1卷积融合通道特征

#         y = y.transpose(-1, -2).unsqueeze(-1)  # 交换维度并在最后添加维度，得到形状为[B, C, 1, 1]
#         y = self.sigmoid(y)  # 对卷积结果应用Sigmoid激活函数，得到通道注意力权重
#         return x * y.expand_as(x)  # 将注意力权重应用到输入上，逐元素相乘，返回加权后的特征图

# # 示例测试
# if __name__ == "__main__":
#     x = torch.randn(4, 64, 32, 32)  # 随机生成一个输入张量，形状为[B, C, H, W] = [4, 64, 32, 32]
#     eca = DSCA_layer(channel=64)  # 创建一个ECA层，通道数为64
#     y = eca(x)  # 将输入张量通过ECA层处理
#     print(y.shape)  # 输出处理后的张量形状，应为[4, 64, 32, 32]


import torch
from torch import nn

class DECA_layer(nn.Module):
    def __init__(self, channel, k_size=3, dilation=2):
        super(DECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出形状为[B, C, 1, 1]

        # 使用空洞卷积替换普通1D卷积
        self.conv = nn.Conv1d(
            in_channels=1,  # 输入通道数为1，因为每次处理一个通道
            out_channels=1,  # 输出通道数为1
            kernel_size=k_size,  # 卷积核大小
            padding=((k_size - 1) // 2) * dilation,  # 空洞卷积的padding计算，保持特征图大小不变
            dilation=dilation,  # 空洞卷积的膨胀系数
            bias=False  # 不使用偏置
        )

        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成通道注意力权重

    def forward(self, x):
        y = self.avg_pool(x)  # 对输入进行全局平均池化，得到形状为[B, C, 1, 1]的张量
        y = y.squeeze(-1).transpose(-1, -2)  # 去掉最后一个维度并交换最后两个维度，得到形状为[B, 1, C]
        y = self.conv(y)  # 对每个通道的特征进行卷积操作
        y = y.transpose(-1, -2).unsqueeze(-1)  # 交换维度并在最后添加维度，得到形状为[B, C, 1, 1]
        y = self.sigmoid(y)  # 对卷积结果应用Sigmoid激活函数，得到通道注意力权重
        return x * y.expand_as(x)  # 将注意力权重应用到输入上，逐元素相乘，返回加权后的特征图

# 示例用法
if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # 随机生成一个输入张量，形状为[B, C, H, W] = [4, 64, 32, 32]
    eca = DECA_layer(channel=64)  # 创建一个ECA层，通道数为64
    y = eca(x)  # 将输入张量通过ECA层处理
    print(y.shape)  # 输出处理后的张量形状，应为[4, 64, 32, 32]

