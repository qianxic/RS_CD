import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math
import os

import functools
from einops import rearrange

import timm

import traceback
import numpy as np


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'plateau':
        # 自适应衰减学习率
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, gpu_ids=None):
    """定义变化检测生成器网络"""
    
    if gpu_ids is None:
        gpu_ids = [0]
        
    # 获取网络类型
    net_type = args.net_G
    
    # 在此基础上创建对应类型的网络
    if net_type == 'vit_base_patch16':
        n_class = args.n_class
        net = VisionTransformer(
            embed_dim=args.embed_dim if hasattr(args, 'embed_dim') else 768,
            depth=args.depth if hasattr(args, 'depth') else 12,
            num_heads=args.num_heads if hasattr(args, 'num_heads') else 12,
            mlp_ratio=args.mlp_ratio if hasattr(args, 'mlp_ratio') else 4.0,
            output_nc=n_class,
            img_size=args.img_size if hasattr(args, 'img_size') else 512,
            vit_img_size=args.vit_img_size if hasattr(args, 'vit_img_size') else 448,
            patch_size=16,
            input_nc=3,
            attn_sparsity=args.attn_sparsity if hasattr(args, 'attn_sparsity') else 0.1,
            fusion_sparsity=args.fusion_sparsity if hasattr(args, 'fusion_sparsity') else 0.2
        )
        print("使用Vision Transformer模型进行变化检测")
    else:
        raise NotImplementedError('网络模型 [%s] 未定义' % net_type)
    
    # 将网络移动到GPU
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        
    # 确保所有参数都开启梯度计算
    for param in net.parameters():
        param.requires_grad = True
        
    return net


###############################################################################
# Main Network Classes
###############################################################################


# 边缘检测辅助任务
class EdgeDetection(nn.Module):
    """
    简化版边缘检测模块
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(EdgeDetection, self).__init__()
        
        # 简化卷积结构
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, C, H, W]
            
        返回:
            边缘检测结果 [B, 1, H, W]
        """
        return torch.sigmoid(self.conv(x))


class VisionTransformer(nn.Module):
    """
    基于ViT的变化检测网络
    使用ViT提取特征，Transformer编解码器融合时序信息，差分计算得到变化掩码
    包含稀疏注意力机制和增强的特征融合
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, 
                 output_nc=2, img_size=512, vit_img_size=448, patch_size=16, input_nc=3,
                 attn_sparsity=0.1, fusion_sparsity=0.2):
        super(VisionTransformer, self).__init__()
        
        # 存储初始化参数信息
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.vit_img_size = vit_img_size  # ViT模型的输入尺寸
        self.depth = depth
        self.num_heads = num_heads
        self.output_nc = output_nc
        self.attn_sparsity = attn_sparsity  # 存储注意力稀疏度参数
        self.fusion_sparsity = fusion_sparsity  # 存储特征融合稀疏度参数
        
        # Vision Transformer作为特征提取器
        # 使用vit_img_size替代固定值224
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True,
            img_size=self.vit_img_size,  # 使用vit_img_size参数
            num_classes=0,
            in_chans=input_nc
        )
        
        # 计算特征图大小
        self.feature_size = self.vit_img_size // patch_size
        
        # 使用自定义的带稀疏注意力的编码器层
        self.sparse_encoder = SparseTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            sparsity=attn_sparsity,
            num_layers=2
        )
        
        # 使用标准解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2
        )
        
        # 渐进式上采样模块 - 从14x14到256x256，按照图中架构调整通道维度
        # 第一阶段：14x14 -> 28x28 (保持768通道)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
        # 第二阶段：28x28 -> 56x56 (减少到384通道)
        self.upsample2 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        
        # 第三阶段：56x56 -> 112x112 (减少到192通道)
        self.upsample3 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # 第四阶段：112x112 -> 224x224 (减少到96通道)
        self.upsample4 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 最终阶段：224x224 -> 256x256 (减少到48通道)
        self.upsample5 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 维度转换模块 - 48通道到2通道
        self.dim_conv = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, output_nc, kernel_size=1)
        )
        
        # 边缘检测模块
        self.edge_detector = EdgeDetection(
            in_channels=48,  # 使用差异计算后的特征维度
            out_channels=1   # 输出单通道边缘图
        )
        
        # 边缘特征融合模块 - 融合边缘特征与维度转换后的特征
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(output_nc + 1, output_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, img1, img2):
        """
        模型前向传播
        
        参数:
            img1: 第一时间点图像 [B, C, H, W]
            img2: 第二时间点图像 [B, C, H, W]
            
        返回:
            output: 最终输出的变化检测结果 [B, 2, H, W]
        """
        # 获取原始输入尺寸，用于最终输出
        B, C, H_original, W_original = img1.shape
        
        # 将图像调整为vit_img_size，以匹配ViT预训练模型的期望输入尺寸
        img1_vit = F.interpolate(img1, size=(self.vit_img_size, self.vit_img_size), 
                           mode='bilinear', align_corners=False)
        img2_vit = F.interpolate(img2, size=(self.vit_img_size, self.vit_img_size), 
                           mode='bilinear', align_corners=False)
        
        try:
            # 1. 提取特征
            feat1 = self.extract_features(img1_vit)  # [B, N, 768] 其中N = (vit_img_size/16)^2
            feat2 = self.extract_features(img2_vit)  # [B, N, 768]
            
            # 2. 合并特征并应用稀疏注意力编码器
            combined_feat = torch.cat([feat1, feat2], dim=1)  # [B, 2*N, 768]
            encoded_feat = self.sparse_encoder(combined_feat)  # 应用稀疏注意力 [B, 2*N, 768]
            
            # 3. 将融合特征拆分回两个时相，用于提供Key/Value
            encoded_feat1, encoded_feat2 = torch.chunk(encoded_feat, 2, dim=1)  # 各 [B, N, 768]
            
            # 4. 分别使用Transformer解码器处理
            # 合并特征作为Query (tgt)，单时相特征作为Key/Value (memory)
            decoded_feat1 = self.transformer_decoder(combined_feat, encoded_feat1)  # [B, 2*N, 768]
            decoded_feat2 = self.transformer_decoder(combined_feat, encoded_feat2)  # [B, 2*N, 768]
            
            # 5. 从解码器输出中分割出对应的特征
            decoded_feat1_half1, _ = torch.chunk(decoded_feat1, 2, dim=1)  # [B, N, 768]
            decoded_feat2_half1, _ = torch.chunk(decoded_feat2, 2, dim=1)  # [B, N, 768]
            
            # 6. 将token特征重塑为空间特征图
            # 特征大小现在是vit_img_size/16
            h = w = self.feature_size  # 例如：448/16 = 28
            feat_map1 = decoded_feat1_half1.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2)  # [B, 768, h, w]
            feat_map2 = decoded_feat2_half1.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2)  # [B, 768, h, w]
            
            # 7. 渐进式上采样 - 从小尺寸逐步上采样到输入图像大小
            # 根据特征图大小调整上采样策略
            
            # 第一阶段：h×w -> 2h×2w (保持768通道)
            x1_a = F.interpolate(feat_map1, size=(h*2, w*2), mode='bilinear', align_corners=False)
            x1_a = self.upsample1(x1_a)
            
            x1_b = F.interpolate(feat_map2, size=(h*2, w*2), mode='bilinear', align_corners=False)
            x1_b = self.upsample1(x1_b)
            
            # 第二阶段：2h×2w -> 4h×4w (减少到384通道)
            x2_a = F.interpolate(x1_a, size=(h*4, w*4), mode='bilinear', align_corners=False)
            x2_a = self.upsample2(x2_a)
            
            x2_b = F.interpolate(x1_b, size=(h*4, w*4), mode='bilinear', align_corners=False)
            x2_b = self.upsample2(x2_b)
            
            # 第三阶段：4h×4w -> 8h×8w (减少到192通道)
            x3_a = F.interpolate(x2_a, size=(h*8, w*8), mode='bilinear', align_corners=False)
            x3_a = self.upsample3(x3_a)
            
            x3_b = F.interpolate(x2_b, size=(h*8, w*8), mode='bilinear', align_corners=False)
            x3_b = self.upsample3(x3_b)
            
            # 直接上采样到原始输入大小 (减少到96通道，然后到48通道)
            x4_a = F.interpolate(x3_a, size=(H_original, W_original), mode='bilinear', align_corners=False)
            x4_a = self.upsample4(x4_a)  # [B, 96, H_original, W_original]
            x4_a = self.upsample5(x4_a)  # [B, 48, H_original, W_original]
            
            x4_b = F.interpolate(x3_b, size=(H_original, W_original), mode='bilinear', align_corners=False)
            x4_b = self.upsample4(x4_b)  # [B, 96, H_original, W_original]
            x4_b = self.upsample5(x4_b)  # [B, 48, H_original, W_original]
            
            # 8. 在上采样后计算差异
            diff = torch.abs(x4_a - x4_b)  # [B, 48, H_original, W_original]
            
            # 应用稀疏化 - 只保留重要的差异
            if self.fusion_sparsity > 0:
                # 对每个样本分别处理
                for i in range(B):
                    # 计算阈值
                    flat_diff = diff[i].flatten()
                    k = int(flat_diff.numel() * self.fusion_sparsity)
                    if k > 0:
                        threshold, _ = torch.kthvalue(flat_diff, k)
                        # 将小于阈值的差异置零
                        mask = diff[i] <= threshold
                        diff[i].masked_fill_(mask, 0)
            
            # 9. 生成变化检测图
            change_map = diff  # [B, 48, H_original, W_original]
            
            # 10. 边缘检测 (作为辅助分支)
            edge_map = self.edge_detector(change_map)  # [B, 1, H_original, W_original]
            
            # 11. 维度转换 - 48通道 -> 2通道
            dim_reduced = self.dim_conv(change_map)  # [B, 2, H_original, W_original]
            
            # 12. 边缘特征融合
            output = torch.cat([dim_reduced, edge_map], dim=1)  # [B, 3, H_original, W_original]
            output = self.edge_fusion(output)  # [B, 2, H_original, W_original]
            
            # 返回字典格式的结果
            return {
                'seg': output,          # 分割结果
                'edge': edge_map        # 边缘检测结果
            }
            
        except Exception as e:
            print(f"前向传播出错: {e}")
            traceback.print_exc()  # 打印完整的错误堆栈跟踪
            # 返回dummy输出
            dummy_output = {
                'seg': torch.zeros(B, self.output_nc, H_original, W_original, device=img1.device),
                'edge': torch.zeros(B, 1, H_original, W_original, device=img1.device)
            }
            return dummy_output

    def extract_features(self, x):
        """
        从图像中提取ViT特征
        
        参数:
            x: 输入图像 [B, C, H, W]，H=W=vit_img_size
            
        返回:
            features: 提取的特征 [B, N, D]，其中N=(vit_img_size/16)^2, D=embed_dim
        """
        # 如果图像尺寸不是vit_img_size，调整大小
        if x.shape[-1] != self.vit_img_size or x.shape[-2] != self.vit_img_size:
            x = F.interpolate(x, size=(self.vit_img_size, self.vit_img_size), mode='bilinear', align_corners=False)
            
        try:
            # 使用backbone提取特征
            features = self.backbone.forward_features(x)
            
            # 由于timm模型可能输出不同格式，确保转换为合适的形状
            if features.dim() == 2:
                # 如果输出是 [B, D]，转换为 [B, 1, D]
                features = features.unsqueeze(1)
            elif len(features.shape) == 3 and features.shape[1] == self.feature_size**2 + 1:
                # 如果包含[CLS]标记，去除它 [B, N+1, D] -> [B, N, D]
                features = features[:, 1:, :]
                
            # 确保特征形状正确
            expected_tokens = self.feature_size**2
            if features.shape[1] != expected_tokens:
                print(f"Warning: Expected {expected_tokens} tokens but got {features.shape[1]}")
                # 尝试处理不匹配的情况
                if features.shape[1] > expected_tokens:
                    features = features[:, :expected_tokens, :]
                else:
                    # 不太可能发生，但如果发生，使用填充
                    padding = torch.zeros(features.shape[0], expected_tokens - features.shape[1], 
                                         features.shape[2], device=features.device)
                    features = torch.cat([features, padding], dim=1)
                    
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            traceback.print_exc()
            # 返回全零张量作为备用方案
            return torch.zeros(x.shape[0], self.feature_size**2, self.embed_dim, device=x.device)


# 稀疏注意力实现
class SparseAttention(nn.Module):
    """
    实现稀疏注意力机制
    通过阈值化注意力权重，将小于阈值的权重置零，实现稀疏化
    """
    def __init__(self, embed_dim, num_heads, sparsity=0.1):
        super(SparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity = sparsity  # 稀疏度 - 要置零的注意力权重比例
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        前向传播计算稀疏注意力
        
        参数:
            x: 输入特征 [B, N, D]
            
        返回:
            output: 注意力计算结果 [B, N, D]
        """
        B, N, D = x.shape
        
        # 投影计算Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, num_heads, N, head_dim]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, num_heads, N, N]
        
        # 应用稀疏化 - 根据阈值保留前 (1-sparsity) 的权重
        if self.sparsity > 0:
            # 对每个头部的注意力矩阵分别处理
            for i in range(self.num_heads):
                # 计算每个注意力矩阵的阈值
                head_attn = attn[:, i]  # [B, N, N]
                for j in range(B):
                    # 将注意力矩阵展平并排序
                    flat_attn = head_attn[j].flatten()
                    k = int(flat_attn.numel() * self.sparsity)
                    if k > 0:
                        # 找到第k小的值作为阈值
                        threshold, _ = torch.kthvalue(flat_attn, k)
                        # 将小于阈值的权重置零
                        mask = head_attn[j] <= threshold
                        head_attn[j].masked_fill_(mask, -1e9)  # 使用-1e9替代float('-inf')
        
        # 应用softmax获取注意力权重
        attn = F.softmax(attn, dim=-1)
        
        # 加权聚合V
        output = (attn @ v).transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        
        return output


# 稀疏Transformer编码器层
class SparseTransformerEncoderLayer(nn.Module):
    """
    带有稀疏注意力机制的Transformer编码器层
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, sparsity=0.1):
        super(SparseTransformerEncoderLayer, self).__init__()
        self.sparse_attn = SparseAttention(embed_dim, num_heads, sparsity)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        
    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.sparse_attn(self.norm1(x))
        # 残差连接 + 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


# 稀疏Transformer编码器
class SparseTransformerEncoder(nn.Module):
    """
    由多个带稀疏注意力的编码器层组成的编码器
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, sparsity=0.1, num_layers=2):
        super(SparseTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                sparsity=sparsity
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 添加trunc_normal_函数
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 按照PyTorch原始实现的截断正态分布初始化函数
    with torch.no_grad():
        # 填充正态分布
        tensor.normal_(mean=mean, std=std)
        # 截断到[a, b]范围
        tensor.clamp_(min=a * std + mean, max=b * std + mean)
    return tensor


# Patch Embedding类
class PatchEmbed(nn.Module):
    """将图像分解为非重叠的patch并线性嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, with_pos='learned'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.with_pos = with_pos
        
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if with_pos == 'learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
            
    def forward(self, x):
        # 直接假设输入是img_size大小，不进行检查和调整
        B, C, H, W = x.shape
        
        # [B, C, H, W] -> [B, D, H//P, W//P] -> [B, D, N] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        if self.with_pos == 'learned':
            # 动态调整位置编码以匹配实际patch数
            actual_num_patches = x.size(1)
            if actual_num_patches != self.num_patches:
                # 重新采样位置编码以匹配当前patch数
                pos_embed = self.pos_embed
                if actual_num_patches > self.num_patches:
                    # 插值扩展位置编码
                    pos_embed = F.interpolate(
                        pos_embed.transpose(1, 2).reshape(1, -1, int(np.sqrt(self.num_patches)), int(np.sqrt(self.num_patches))),
                        size=(int(np.sqrt(actual_num_patches)), int(np.sqrt(actual_num_patches))),
                        mode='bilinear', align_corners=False
                    ).flatten(2).transpose(1, 2)
                else:
                    # 降采样位置编码
                    pos_embed = F.interpolate(
                        pos_embed.transpose(1, 2).reshape(1, -1, int(np.sqrt(self.num_patches)), int(np.sqrt(self.num_patches))),
                        size=(int(np.sqrt(actual_num_patches)), int(np.sqrt(actual_num_patches))),
                        mode='bilinear', align_corners=False
                    ).flatten(2).transpose(1, 2)
                
                x = x + pos_embed
            else:
                x = x + self.pos_embed
        
        return x


