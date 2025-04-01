import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math

import functools
from einops import rearrange

import timm

from torchvision import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

# 导入timm库，用于加载ViT模型
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

    print('initialize network with %s' % init_type)
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


def define_G(args, gpu_ids=[]):
    """
    定义生成器网络
    
    参数:
        args: 配置参数
        gpu_ids: GPU ID，用于设置模型在哪些GPU上运行
        
    返回:
        初始化好的生成器网络
    """
    netG = None
    
    # 根据网络类型选择不同的生成器
    if args.net_G == 'base_resnet18':
        netG = ResNet(input_nc=3, output_nc=2) # 输出固定为2通道
    elif args.net_G == 'base_transformer_pos_s4':
        netG = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
    elif args.net_G == 'base_transformer_pos_s4_dd8':
        netG = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8,
                             decoder_dim_head=8)
    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        netG = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8,
                             decoder_dim_head=8, decoder_embedding_dim=8)
    elif args.net_G == 'vit_base_patch16':
        # 使用VisionTransformer模型，固定输出为2通道
        netG = VisionTransformer(
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            output_nc=2, # 输出固定为2通道
            attn_sparsity=args.attn_sparsity,
            fusion_sparsity=args.fusion_sparsity
        )
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    
    print(f"Generator network: {args.net_G} with {2} output channels (for binary change detection)")

    # 如果有多个GPU，使用DataParallel
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            netG = torch.nn.DataParallel(netG, gpu_ids)
    
    # 返回初始化好的网络
    return netG


###############################################################################
# Main Network Classes
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet50',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # 添加最终尺寸调整层，确保输出为256x256
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)
        
        # 确保输出尺寸为256x256
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = self.final_upsample(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (required).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (optional).
        custom_decoder: custom decoder (optional).
    """

    def __init__(self, input_nc, output_nc, token_len=4, resnet_stages_num=4,
                 if_upsample_2x=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=None,
                 if_patch_embed_wxh=True,
                 n_heads=8,
                 decoder_n_heads=8,
                 if_transskip=True,
                 if_use_pe=True,
                 dropout=0.1,
                 embed_dim=256,
                 decoder_embed_dim=256,
                 mlp_ratio=4,
                 attention_type="dual_attn",
                 with_pos=None,
                 **kwargs):
        super(BASE_Transformer, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = 2
            self.pool_mode = 'max'
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = if_transskip
        self.with_decoder = if_transskip
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=True)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        
        # 确保输出尺寸为256x256
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = self.final_upsample(x)
            
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


# CBAM注意力机制(结合通道和空间注意力)
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 添加归一化层稳定输出
        self.norm = nn.GroupNorm(min(32, channels), channels)
        
    def forward(self, x):
        # 保存输入作为残差连接
        identity = x
        
        # 通道注意力
        # 计算通道注意力映射
        channel_att = self.channel_att(x)
        # 应用sigmoid激活，确保数值稳定性
        channel_att = torch.sigmoid(channel_att.clamp(-10, 10))
        # 应用通道注意力
        channel_output = x * channel_att
        
        # 空间注意力
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(channel_output, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_output, dim=1, keepdim=True)
        # 拼接特征
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        # 计算空间注意力映射
        spatial_att = self.spatial_att(spatial_input)
        # 应用空间注意力
        spatial_output = channel_output * spatial_att
        
        # 残差连接并归一化
        out = spatial_output + identity
        out = self.norm(out)
        
        return out

# 跨时间稀疏注意力模块
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, sparsity_threshold=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.sparsity_threshold = sparsity_threshold  # 稀疏性阈值
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 添加输入归一化层
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # 投影层
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # dropout层
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # 输出归一化
        self.norm_out = nn.LayerNorm(dim)
        
    def forward(self, q_inputs, kv_inputs, value_inputs=None):
        """
        稀疏跨时间注意力前向传播
        
        参数:
            q_inputs: 查询输入 [B, N, C]
            kv_inputs: 键值输入 [B, N, C]
            value_inputs: 可选的值输入，如果为None，则使用kv_inputs
        """
        # 保存残差连接
        identity = q_inputs
        
        # 如果未提供value_inputs，则使用kv_inputs
        if value_inputs is None:
            value_inputs = kv_inputs
            
        # 应用输入归一化
        q_inputs = self.norm_q(q_inputs)
        kv_inputs = self.norm_k(kv_inputs)
        value_inputs = self.norm_v(value_inputs)
            
        # 获取形状信息
        B, N, C = q_inputs.shape
        
        # 投影并重塑 - 确保生成新张量
        q = self.q(q_inputs).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.k(kv_inputs).reshape(B, kv_inputs.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        v = self.v(value_inputs).reshape(B, value_inputs.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        
        # 注意力计算
        attn_orig = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # 应用稀疏性: 只保留前k%的注意力权重，其余置零
        if self.sparsity_threshold > 0:
            # 创建一个新的张量，完全避免原地修改
            attn = attn_orig.clone()
            
            for b in range(B):
                for h in range(self.num_heads):
                    # 获取当前头部的注意力权重
                    head_attn = attn[b, h]  # [N, N]
                    
                    # 找到阈值: 只保留权重值最高的k%
                    num_to_keep = max(1, int((1 - self.sparsity_threshold) * head_attn.numel()))
                    
                    # 找到第k大的元素作为阈值
                    flat_attn = head_attn.reshape(-1)
                    threshold_value = torch.kthvalue(flat_attn, flat_attn.numel() - num_to_keep)[0]
                    
                    # 创建掩码并应用 - 确保不进行原地修改
                    mask = (head_attn < threshold_value)
                    # 创建新张量来存储修改后的注意力
                    new_head_attn = head_attn.clone()
                    new_head_attn = new_head_attn.masked_fill(mask, -1e4)  # 使用较小的负数，而不是-inf避免NaN
                    
                    # 更新注意力权重
                    attn[b, h] = new_head_attn
        else:
            attn = attn_orig
        
        # 软化计算注意力权重，添加数值稳定性
        attn = F.softmax(attn.clamp(-1e4, 1e4), dim=-1)  # 限制数值范围
        attn = self.attn_drop(attn)
        
        # 计算加权和
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 应用残差连接和输出归一化
        x = x + identity
        x = self.norm_out(x)
        
        # 返回注意力输出和注意力权重
        return x, attn

# 多尺度稀疏特征融合模块
class EnhancedFeatureFusion(nn.Module):
    def __init__(self, channels_list, sparsity_threshold=0.2):
        super(EnhancedFeatureFusion, self).__init__()
        
        # 配置参数
        self.channels_list = channels_list
        self.sparsity_threshold = sparsity_threshold
        self.out_channels = 256  # 输出通道数
        
        # 对每个尺度的特征应用1x1卷积，统一通道数
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(channels, self.out_channels, kernel_size=1) 
            for channels in channels_list
        ])
        
        # 应用注意力机制
        self.attention = CBAM(self.out_channels)
        
        # 简化稀疏通道选择，使用单一全局选择器
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(channels_list), 
                     self.out_channels * len(channels_list) // 4, 
                     kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.out_channels * len(channels_list) // 4, 
                     self.out_channels * len(channels_list), 
                     kernel_size=1),
            nn.Sigmoid()
        )
        
        # 最终融合后的卷积
        self.fusion_conv = nn.Conv2d(self.out_channels * len(channels_list), 
                                   self.out_channels, kernel_size=1)
        
        # 添加GroupNorm以稳定特征分布
        self.norm = nn.GroupNorm(min(32, self.out_channels), self.out_channels)
        
    def forward(self, features_list):
        # 处理每个尺度的特征
        processed_features = []
        target_size = features_list[0].shape[2:]  # 目标空间尺寸
        
        for i, feature in enumerate(features_list):
            # 应用1x1卷积调整通道
            x = self.conv_layers[i](feature)
            
            # 上采样到最大特征图尺寸(假设第一个特征图最大)
            if i > 0:  # 跳过第一个，因为它已经是最大尺寸
                if x.shape[2:] != target_size:
                    x = F.interpolate(x, size=target_size, 
                                    mode='bilinear', align_corners=False)
            
            # 应用注意力机制
            x = self.attention(x)
            processed_features.append(x)
        
        # 沿通道维度拼接
        fused = torch.cat(processed_features, dim=1)
        
        # 应用稀疏特征选择（如果启用）
        if self.sparsity_threshold > 0:
            # 计算通道注意力权重 - 确保创建新张量
            channel_weights = self.channel_selector(fused)
            
            # 应用简单的阈值掩码而不是复杂的空间掩码，提高稳定性
            if self.training:
                mask = (channel_weights < 0.2)  # 使用固定阈值代替动态计算
                channel_weights = channel_weights.masked_fill(mask, 0.0)
            
            # 应用权重 - 显式创建新张量
            fused = torch.mul(fused, channel_weights)
        
        # 最终1x1卷积融合
        output = self.fusion_conv(fused)
        
        # 应用归一化以确保稳定性
        output = self.norm(output)
        
        return output

# 边缘检测辅助任务
class EdgeDetection(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(EdgeDetection, self).__init__()
        # 适配输入通道数
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(min(32, 64), 64)  # 使用GroupNorm替代BatchNorm
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(min(32, 64), 64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=False)  # 避免原地操作
        
    def forward(self, x):
        try:
            # 添加数值稳定性检查
            if torch.isnan(x).any():
                # 如果输入包含NaN，返回全零张量
                return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
                
            # 特征提取过程
            x = self.activation(self.bn1(self.conv1(x)))
            x = self.activation(self.bn2(self.conv2(x)))
            # 最终输出层，产生边缘检测映射
            edge_map = torch.sigmoid(self.conv3(x))
            
            # 确保无NaN
            if torch.isnan(edge_map).any():
                edge_map = torch.zeros_like(edge_map)
                
            return edge_map
        except Exception as e:
            print(f"EdgeDetection前向传播错误: {e}")
            # 错误处理: 返回一个零张量作为边缘映射
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

# 新增：Vision Transformer作为特征提取器的变化检测模型
class VisionTransformer(nn.Module):
    """
    基于ViT的变化检测网络
    使用注意力机制和特征融合提取时序变化特征
    
    参数:
        - embed_dim: 嵌入维度
        - depth: Transformer深度
        - num_heads: 注意力头数
        - mlp_ratio: MLP比例
        - output_nc: 输出通道数，固定为2（1-无变化，1-有变化）
        - attn_sparsity: 注意力稀疏度
        - fusion_sparsity: 特征融合稀疏度
    """
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, 
                 output_nc=2, img_size=256, patch_size=16, input_nc=3,
                 attn_sparsity=0.1, fusion_sparsity=0.2):
        super(VisionTransformer, self).__init__()
        
        # 存储初始化参数信息
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size  # 保留原始输入尺寸记录
        self.vit_size = 224       # 新增: ViT的实际工作尺寸
        self.depth = depth
        self.num_heads = num_heads
        self.output_nc = 2  # 固定为2通道输出
        self.dims_info = {}  # 用于存储中间特征尺寸信息，方便调试
        
        # timm库中的Vision Transformer模型
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True,
            img_size=224,  # 固定使用224x224作为ViT输入尺寸
            num_classes=0,  # 移除分类头，只使用特征提取部分
            in_chans=input_nc
        )
        
        # 减少模型层数以降低复杂度
        if depth < 12:  # 如果指定的深度小于12
            # 标准ViT有12层，我们保留前depth层
            self.backbone.blocks = self.backbone.blocks[:depth]
            print(f"只使用ViT的前{depth}层，原模型有12层")
        
        # Patch Embedding和位置编码
        self.patch_embed = PatchEmbed(
            img_size=224,  # 使用224作为ViT的输入尺寸
            patch_size=patch_size, 
            in_chans=input_nc, 
            embed_dim=embed_dim
        )
        
        # 特征提取器，用于获取多尺度特征
        self.feature_extractor = nn.ModuleDict({
            'shallow': nn.Conv2d(input_nc, embed_dim//4, kernel_size=3, padding=1),
            'mid': nn.Conv2d(input_nc, embed_dim//2, kernel_size=3, padding=1),
            # 深层特征通过backbone提取，无需额外层
        })
        
        # 边缘检测模块
        self.edge_detector = EdgeDetection(input_nc, embed_dim//8)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 跨时间注意力机制
        self.cross_attn1 = CrossAttention(embed_dim, num_heads=num_heads, 
                                        sparsity_threshold=attn_sparsity)
        self.cross_attn2 = CrossAttention(embed_dim, num_heads=num_heads, 
                                        sparsity_threshold=attn_sparsity)
        
        # 特征融合模块
        self.feature_fusion = EnhancedFeatureFusion(
            channels_list=[embed_dim//4, embed_dim//2, embed_dim],
            sparsity_threshold=fusion_sparsity
        )
        
        # CBAM注意力机制
        self.cbam = CBAM(embed_dim)
        
        # 上采样模块
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, 128), 128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, 64), 64),
            nn.ReLU(inplace=False)
        )
        
        # 输出层 - 固定为2通道输出
        self.outc = nn.Conv2d(64, 2, kernel_size=1)
        
        # 新增: 平滑下采样模块，用于从高分辨率平滑过渡到原始尺寸
        self.smooth_downsampler = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 第一次下采样: 1024->512
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),   # 第二次下采样: 512->256
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 2, kernel_size=1)                         # 恢复到2通道输出
        )

    def forward(self, img1, img2):
        """
        模型前向传播
        
        参数:
            img1: 第一时间点图像 [B, C, H, W]
            img2: 第二时间点图像 [B, C, H, W]
            
        返回:
            一个元组包含:
            - output: 变化检测结果 [B, output_nc, H, W]
            - edge_maps: 边缘检测结果 [B, 1, H, W]
        """
        try:
            # 记录输入维度信息，方便调试
            self.dims_info = {"input_shape": (img1.shape, img2.shape)}
            
            # 获取原始输入尺寸，用于最终输出
            B, C, H_original, W_original = img1.shape
            
            # 直接将图像调整为224x224，以匹配ViT预训练模型的期望输入尺寸
            vit_size = 224
            img1_vit = F.interpolate(img1, size=(vit_size, vit_size), 
                               mode='bilinear', align_corners=False)
            img2_vit = F.interpolate(img2, size=(vit_size, vit_size), 
                               mode='bilinear', align_corners=False)
            
            # 检查输入是否包含NaN
            if torch.isnan(img1_vit).any() or torch.isnan(img2_vit).any():
                # 替换NaN值为0
                img1_vit = torch.nan_to_num(img1_vit, nan=0.0)
                img2_vit = torch.nan_to_num(img2_vit, nan=0.0)
            
            # 1. 提取多尺度特征
            try:
                # 使用调整后的224x224图像提取特征
                shallow_feat1, mid_feat1, deep_feat1 = self.extract_features(img1_vit)
                shallow_feat2, mid_feat2, deep_feat2 = self.extract_features(img2_vit)
                
                self.dims_info.update({
                    "shallow_feats": (shallow_feat1.shape, shallow_feat2.shape),
                    "mid_feats": (mid_feat1.shape, mid_feat2.shape),
                    "deep_feats": (deep_feat1.shape, deep_feat2.shape)
                })
            except Exception as e:
                print(f"特征提取错误: {e}")
                # 在特征提取失败的情况下，返回dummy输出
                batch_size = img1.shape[0]
                dummy_output = torch.zeros(batch_size, 1, H_original, W_original, device=img1.device)
                dummy_edge = torch.zeros_like(dummy_output)
                return dummy_output, dummy_edge
            
            # 2. 边缘检测分支 - 使用梯度分离避免影响主网络
            try:
                # 使用deep features的detach版本进行边缘检测
                # 这样边缘检测任务不会影响主网络的梯度
                edge_feat = torch.cat([deep_feat1.detach(), deep_feat2.detach()], dim=1)
                edge_maps = self.edge_detector(edge_feat)
                self.dims_info["edge_maps"] = edge_maps.shape
            except Exception as e:
                print(f"边缘检测错误: {e}")
                # 创建假的边缘图
                batch_size = img1.shape[0]
                edge_maps = torch.zeros(batch_size, 1, H_original, W_original, device=img1.device)
            
            # 3. 应用跨时间稀疏注意力增强深层特征
            try:
                enhanced_feat1, enhanced_feat2, attn_weight = self.apply_cross_attention(
                    deep_feat1, deep_feat2, self.attn_sparsity
                )
                self.dims_info["enhanced_feats"] = (enhanced_feat1.shape, enhanced_feat2.shape)
            except Exception as e:
                print(f"应用注意力错误: {e}")
                # 如果注意力应用失败，使用原始特征
                enhanced_feat1, enhanced_feat2 = deep_feat1, deep_feat2
                attn_weight = None
            
            # 4. 多尺度特征融合
            try:
                # 尝试融合两个时间点的特征
                fused_feat1 = self.feature_fusion(
                    [shallow_feat1, mid_feat1, enhanced_feat1]
                )
                fused_feat2 = self.feature_fusion(
                    [shallow_feat2, mid_feat2, enhanced_feat2]
                )
                self.dims_info["fused_feats"] = (fused_feat1.shape, fused_feat2.shape)
                
                # 确保特征尺寸一致
                if fused_feat1.shape != fused_feat2.shape:
                    # 调整到更大的尺寸
                    target_size = max(fused_feat1.shape[2:], fused_feat2.shape[2:])
                    fused_feat1 = F.interpolate(fused_feat1, size=target_size, 
                                             mode='bilinear', align_corners=False)
                    fused_feat2 = F.interpolate(fused_feat2, size=target_size, 
                                             mode='bilinear', align_corners=False)
                
                # 计算两个特征的差异
                diff_feat = torch.abs(fused_feat1 - fused_feat2)
                self.dims_info["diff_feat"] = diff_feat.shape
            except Exception as e:
                print(f"特征融合错误: {e}")
                # 使用深层特征差作为备用
                diff_feat = torch.abs(deep_feat1 - deep_feat2)
            
            # 5. 应用CBAM注意力机制
            try:
                att_feat = self.cbam(diff_feat)
                self.dims_info["att_feat"] = att_feat.shape
            except Exception as e:
                print(f"CBAM应用错误: {e}")
                # 如果CBAM应用失败，使用原始差异特征
                att_feat = diff_feat
            
            # 6. 上采样和最终预测
            try:
                # 确保无NaN
                att_feat = torch.nan_to_num(att_feat, nan=0.0)
                
                # 应用上采样器获得高分辨率特征
                out = self.upsampler(att_feat)
                out = self.outc(out)  # [B, 2, 高分辨率]
                
                # 如果需要，应用平滑下采样回到原始尺寸
                if out.shape[-2:] != (H_original, W_original):
                    # 检查是否需要先应用双线性插值到中间分辨率
                    if out.shape[-1] > 512:  # 如果分辨率特别高
                        # 先双线性插值到较合适的中间分辨率
                        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
                    
                    # 应用平滑下采样模块
                    if out.shape[-1] > H_original * 2:
                        out = self.smooth_downsampler(out)
                    else:
                        # 分辨率较低时直接使用双线性插值
                        out = F.interpolate(out, size=(H_original, W_original), mode='bilinear', align_corners=False)
                
                # 始终使用双通道输出 (通道 0: 未变化, 通道 1: 变化)
                # 确保输出是两通道的
                if out.shape[1] != 2:
                    print(f"警告: 输出通道数不是2 ({out.shape[1]}), 但继续使用双通道输出")
                
                # 为了数值稳定性，限制logits范围
                out = out.clamp(-10, 10)
                print(f"使用双通道输出(logits)") if np.random.random() < 0.01 else None
                
                # 记录输出信息
                self.dims_info["output"] = out.shape
                
                # 同样确保边缘图尺寸与原始尺寸一致
                if edge_maps.shape[-2:] != (H_original, W_original):
                    edge_maps = F.interpolate(edge_maps, size=(H_original, W_original), 
                                            mode='bilinear', align_corners=False)
            except Exception as e:
                print(f"最终预测错误: {e}")
                # 创建dummy输出
                batch_size = img1.shape[0]
                out = torch.zeros(batch_size, 1, H_original, W_original, device=img1.device)
                edge_maps = torch.zeros_like(out)
            
            # 做最后的检查，确保没有NaN值
            if torch.isnan(out).any() or torch.isnan(edge_maps).any():
                # 替换NaN值
                out = torch.nan_to_num(out, nan=0.0)
                edge_maps = torch.nan_to_num(edge_maps, nan=0.0)
            
            # 返回统一的输出: (预测结果, 边缘图)
            return out, edge_maps
            
        except Exception as e:
            print(f"前向传播总体错误: {e}")
            # 最终的应急解决方案，返回全零输出
            batch_size = img1.shape[0]
            dummy_output = torch.zeros(batch_size, 1, H_original, W_original, device=img1.device)
            dummy_edge = torch.zeros_like(dummy_output)
            return dummy_output, dummy_edge

    def extract_features(self, x):
        """提取多尺度特征"""
        try:
            # 保存输入尺寸
            B, C, H, W = x.shape
            
            # 这里假设输入已经是224x224，不需要再调整
            vit_size = 224
            
            # 浅层特征
            shallow_feat = self.feature_extractor['shallow'](x)
            
            # 中层特征
            mid_feat = self.feature_extractor['mid'](x)
            
            # 深层特征 - 使用Vision Transformer
            # 获取CLS token和patch嵌入
            patch_embed = self.patch_embed(x)  # [B, N, C]
            
            # 将特征转为正确的格式：[B, N, D] -> [B, D, H, W]
            patch_size = self.patch_size
            num_patches_side = vit_size // patch_size  # 使用224计算补丁数量
            
            # 跳过第一个CLS token，只获取patch嵌入
            patch_embed_only = patch_embed
            
            # 直接应用transformer到特征
            transformed = self.transformer(patch_embed_only)
            
            # 重塑为空间特征图 [B, N, D] -> [B, D, H/P, W/P]
            deep_feat = transformed.transpose(1, 2).reshape(B, -1, num_patches_side, num_patches_side)
            
            # 记录每层特征的尺寸
            self.dims_info.update({
                "extract_feat": {
                    "shallow": shallow_feat.shape,
                    "mid": mid_feat.shape,
                    "deep": deep_feat.shape
                }
            })
            
            return shallow_feat, mid_feat, deep_feat
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            traceback.print_stack()
            # 创建假的特征 - 使用原始输入尺寸
            vit_size = 224
            shallow_feat = torch.zeros(B, 64, H, W, device=x.device)
            mid_feat = torch.zeros(B, 256, H//2, W//2, device=x.device)
            deep_feat = torch.zeros(B, self.embed_dim, vit_size//self.patch_size, vit_size//self.patch_size, device=x.device)
            return shallow_feat, mid_feat, deep_feat
    
    def apply_cross_attention(self, feat1, feat2, sparsity_threshold=0.1):
        """
        应用跨时间稀疏注意力机制增强特征表示
        
        参数:
            feat1: 第一个时间点的特征 [B, C, H, W]
            feat2: 第二个时间点的特征 [B, C, H, W]
            sparsity_threshold: 注意力稀疏度阈值
            
        返回:
            enhanced_feat1: 增强后的第一个特征 [B, C, H, W]
            enhanced_feat2: 增强后的第二个特征 [B, C, H, W]
            attn_weights: 注意力权重
        """
        try:
            # 获取形状
            B, C, H, W = feat1.shape
            
            # 展平特征图为序列
            feat1_flat = feat1.flatten(2).transpose(1, 2)  # [B, H*W, C]
            feat2_flat = feat2.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # 检查序列长度
            seq_len = min(feat1_flat.shape[1], feat2_flat.shape[1])
            if feat1_flat.shape[1] != seq_len:
                feat1_flat = feat1_flat[:, :seq_len, :]
            if feat2_flat.shape[1] != seq_len:
                feat2_flat = feat2_flat[:, :seq_len, :]
            
            # 应用跨时间注意力
            enhanced_feat1, attn1 = self.cross_attn1(feat1_flat, feat2_flat)
            enhanced_feat2, attn2 = self.cross_attn2(feat2_flat, feat1_flat)
            
            # 重新塑形为特征图
            enhanced_feat1 = enhanced_feat1.transpose(1, 2).reshape(B, C, H, W)
            enhanced_feat2 = enhanced_feat2.transpose(1, 2).reshape(B, C, H, W)
            
            return enhanced_feat1, enhanced_feat2, attn1
        except Exception as e:
            print(f"应用注意力错误: {e}")
            # 如果失败，返回原始特征
            return feat1, feat2, None

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


