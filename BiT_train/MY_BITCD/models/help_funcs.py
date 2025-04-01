import torch
import torch.nn.functional as F
from einops import rearrange  # einops库用于张量的重排操作
from torch import nn


class TwoLayerConv2d(nn.Sequential):
    """
    双层卷积网络
    实现两个连续的卷积层，中间带有批归一化和ReLU激活函数
    在模型中用于对特征图进行处理，通常作为最终分类器
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),  # 第一层卷积，保持通道数不变
                         nn.BatchNorm2d(in_channels),  # 批归一化
                         nn.ReLU(),  # ReLU激活函数
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)  # 第二层卷积，改变通道数
                         )


class Residual(nn.Module):
    """
    残差连接模块
    将输入与函数输出相加，实现残差连接，有助于训练更深的网络
    用于transformer中的自注意力或前馈网络模块
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 要应用的函数
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x  # 残差连接: 输出 = 函数(x) + x


class Residual2(nn.Module):
    """
    双输入残差连接模块
    类似Residual，但适用于接收两个输入（如query和memory）的函数
    主要用于transformer解码器中的交叉注意力
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 要应用的函数
    
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x  # 残差连接: 输出 = 函数(x, x2) + x


class PreNorm(nn.Module):
    """
    预归一化模块
    在应用函数前先进行层归一化，是transformer架构的重要组成部分
    有助于训练稳定性
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn  # 要应用的函数
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 先归一化，再应用函数


class PreNorm2(nn.Module):
    """
    双输入预归一化模块
    类似PreNorm，但适用于有两个输入的函数
    用于transformer解码器中的交叉注意力
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn  # 要应用的函数
    
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)  # 对两个输入都进行归一化，再应用函数


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    Transformer中的FFN部分，由两个线性层组成，中间有GELU激活和Dropout
    用于在自注意力后进一步处理特征
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 第一个线性层，扩展维度
            nn.GELU(),  # GELU激活函数，比ReLU更平滑
            nn.Dropout(dropout),  # dropout正则化
            nn.Linear(hidden_dim, dim),  # 第二个线性层，恢复原始维度
            nn.Dropout(dropout)  # 再次dropout
        )
    
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    """
    交叉注意力模块
    实现transformer中的交叉注意力机制，用于解码器中query和memory之间的注意力计算
    在变化检测中用于融合两个时相特征
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads  # 多头注意力的总维度
        self.heads = heads  # 注意力头数
        self.scale = dim ** -0.5  # 缩放因子，防止点积过大
        self.softmax = softmax  # 是否使用softmax归一化注意力权重

        # 投影矩阵
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # query投影
        self.to_k = nn.Linear(dim, inner_dim, bias=False)  # key投影
        self.to_v = nn.Linear(dim, inner_dim, bias=False)  # value投影

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 将多头拼接结果投影回原始维度
            nn.Dropout(dropout)  # dropout正则化
        )

    def forward(self, x, m, mask = None):
        """
        x: query张量
        m: key/value张量（memory）
        mask: 可选的掩码，用于屏蔽某些位置
        """
        b, n, _, h = *x.shape, self.heads  # 批大小，序列长度，嵌入维度，头数
        
        # 投影并分离多头
        q = self.to_q(x)  # [b, n, h*d]
        k = self.to_k(m)  # [b, m, h*d]
        v = self.to_v(m)  # [b, m, h*d]

        # 重排为多头形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])  # [b, h, n/m, d]

        # 计算注意力分数 (点积注意力)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # [b, h, n, m]
        mask_value = -torch.finfo(dots.dtype).max  # 掩码值（非常大的负数）

        # 应用掩码（如果提供）
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 应用softmax获得注意力权重
        if self.softmax:
            attn = dots.softmax(dim=-1)  # [b, h, n, m]
        else:
            attn = dots  # 无softmax，使用原始分数

        # 加权汇总
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # [b, h, n, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [b, n, h*d]
        out = self.to_out(out)  # [b, n, dim]

        return out


class Attention(nn.Module):
    """
    自注意力模块
    实现transformer中的自注意力机制，用于编码序列内部的依赖关系
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 多头注意力的总维度
        self.heads = heads  # 注意力头数
        self.scale = dim ** -0.5  # 缩放因子，防止点积过大

        # 联合投影矩阵（同时生成q,k,v）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 一次性生成q,k,v
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 将多头拼接结果投影回原始维度
            nn.Dropout(dropout)  # dropout正则化
        )

    def forward(self, x, mask = None):
        """
        x: 输入序列张量
        mask: 可选的掩码，用于屏蔽某些位置
        """
        b, n, _, h = *x.shape, self.heads  # 批大小，序列长度，嵌入维度，头数
        
        # 一次性生成qkv并分割
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 3个 [b, n, h*d]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 3个 [b, h, n, d]

        # 计算注意力分数 (点积注意力)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # [b, h, n, n]
        mask_value = -torch.finfo(dots.dtype).max  # 掩码值（非常大的负数）

        # 应用掩码（如果提供）
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 应用softmax获得注意力权重
        attn = dots.softmax(dim=-1)  # [b, h, n, n]

        # 加权汇总
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # [b, h, n, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [b, n, h*d]
        out = self.to_out(out)  # [b, n, dim]
        
        return out


class Transformer(nn.Module):
    """
    Transformer编码器
    实现标准的Transformer编码器，由多层自注意力和前馈网络组成
    在变化检测中用于处理令牌化后的特征
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # 创建多层编码器
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 自注意力模块（带残差和预归一化）
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                # 前馈网络模块（带残差和预归一化）
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    
    def forward(self, x, mask = None):
        """
        x: 输入序列张量
        mask: 可选的掩码
        """
        # 依次通过每一层
        for attn, ff in self.layers:
            x = attn(x, mask = mask)  # 自注意力
            x = ff(x)  # 前馈网络
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    实现带有交叉注意力的Transformer解码器
    在变化检测中用于融合不同时相的特征
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        # 创建多层解码器
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 交叉注意力模块（带残差和预归一化）
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                # 前馈网络模块（带残差和预归一化）
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    
    def forward(self, x, m, mask = None):
        """
        x: 目标（query）序列
        m: 记忆（memory/key-value）序列
        mask: 可选的掩码
        """
        # 依次通过每一层
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)  # 交叉注意力
            x = ff(x)  # 前馈网络
        return x


