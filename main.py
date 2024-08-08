import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class pre_process(nn.Module) :
    def __init__(self, image_size, patch_size, patch_dim, dim):
        """
        image_size:输入图像的大小,例如224
        patch_size:每个patch的大小,例如16
        patch_dim: 每个patch的特征维度,例如768
        dim: 模型的隐藏层维度,例如768
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        # 计算图像中patch的数量，并赋值给类属性 self.patch_num
        self.patch_num = (image_size // patch_size) ** 2
        # 初始化一个线性层 self.linear_embedding,用于将每个patch的特征维度从patch_dim映射到dim
        self.linear_embedding = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1,self.patch_num+1, self.dim)) # 使用广播
        self.CLS_token = nn.Parameter(torch.randn(1, 1, self.dim)) # 别忘了维度要和(B,L,C)对齐

    def forward(self, x):
        """
        输入的x的数组表示为(B,C,H,W)，需要将它划分为(B,L,C)
        输入形状- B:批次大小  C:通道数  H:高度 W:宽度
        输出形状- B:批次大小  L:patch的个数,等于(H/p1)*(W/p2)  C:每个patch的像素值数量,等于 p1*p2*channels
        """
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size) # (B, L, C)
        """

        """
        x = self.linear_embedding(x)
        b, l, c = x.shape
        CLS_token = repeat(self.CLS_token, '1 1 d -> b 1 d', b=b) # 位置编码复制 B 份
