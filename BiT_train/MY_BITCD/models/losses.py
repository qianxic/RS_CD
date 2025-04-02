import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#损失函数定义，将被主要运用于训练过程。
def cross_entropy(logits, label, class_weights=None, reduction='mean'):
    """
    交叉熵损失函数，增加了错误处理和标签清理
    
    参数:
        logits: 模型输出的logits [B, C, H, W]
        label: 真实标签 [B, H, W]
        class_weights: 类别权重 [C]
        reduction: 损失汇总方式，'mean', 'sum', 'none'
    """
    # 处理输入
    if logits.shape[1] > 1:  # 多类别情况
        # 如果标签不是long类型，转换为long
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long().to(logits.device)
        elif label.dtype != torch.int64:
            label = label.long()
            
        # 确保标签值在有效范围内
        if label.max() >= logits.shape[1] or label.min() < 0:
            print(f"警告: 标签值超出范围! min={label.min().item()}, max={label.max().item()}, 类别数={logits.shape[1]}")
            # 裁剪标签到有效范围
            label = torch.clamp(label, 0, logits.shape[1]-1)
        
        try:
            # 计算交叉熵损失
            if class_weights is not None:
                # 确保权重在正确的设备上
                if not class_weights.device == logits.device:
                    class_weights = class_weights.to(logits.device)
                    
                criterion = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
            else:
                criterion = nn.CrossEntropyLoss(reduction=reduction)
                
            loss = criterion(logits, label)
            return loss
        except Exception as e:
            print(f"交叉熵损失计算错误: {e}")
            print(f"logits shape: {logits.shape}, label shape: {label.shape}")
            print(f"label min: {label.min().item()}, max: {label.max().item()}")
            # 回退到简单的均方误差损失
            pred = torch.softmax(logits, dim=1)[:, 1]  # 取第二个通道作为预测
            label_one_hot = F.one_hot(label, num_classes=logits.shape[1]).float()
            label_one_hot = label_one_hot.permute(0, 3, 1, 2)
            mse_loss = F.mse_loss(pred, label_one_hot[:, 1])
            return mse_loss
    else:  # 单类别情况，使用二元交叉熵
        pred = torch.sigmoid(logits)
        if label.dtype != pred.dtype:
            label = label.float()
            
        criterion = nn.BCELoss(reduction=reduction)
        loss = criterion(pred, label)
        return loss

def binary_ce(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    二值交叉熵损失函数
    """
    target = target.float() # 确保目标是浮点类型
    
    # 处理目标张量维度，确保它是3D张量 [batch_size, height, width]
    if target.dim() == 5:  # [batch_size, 1, height, width, channels]
        print(f"二分类损失处理5D标签: {target.shape}")
        target = target[:, 0, :, :, 0]  # 取第一个通道和第一个深度通道
    elif target.dim() == 4 and target.shape[-1] == 3:  # [batch_size, height, width, 3]
        print(f"二分类损失处理4D RGB标签: {target.shape}")
        target = target[:, :, :, 0]  # 只保留第一个通道
    elif target.dim() == 4:  # [batch_size, 1, height, width]
        target = torch.squeeze(target, dim=1)
    
    # 确保输入和目标的空间维度匹配
    if input.shape[-2:] != target.shape[-2:]:
        print(f"二分类损失调整输入尺寸从 {input.shape[-2:]} 到 {target.shape[-2:]}")
        # 使用目标的高度和宽度进行插值
        input = F.interpolate(input, size=(target.shape[-2], target.shape[-1]), 
                             mode='bilinear', align_corners=True)
    
    return F.binary_cross_entropy_with_logits(input=input, target=target, 
                                            weight=weight, reduction=reduction)
