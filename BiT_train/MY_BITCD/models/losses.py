import torch
import torch.nn.functional as F

#损失函数定义，将被主要运用于训练过程。
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    
    # 处理目标张量维度，确保它是3D张量 [batch_size, height, width]
    if target.dim() == 5:  # [batch_size, 1, height, width, channels]
        print(f"损失函数处理5D标签: {target.shape}")
        target = target[:, 0, :, :, 0]  # 取第一个通道和第一个深度通道
    elif target.dim() == 4 and target.shape[-1] == 3:  # [batch_size, height, width, 3]
        print(f"损失函数处理4D RGB标签: {target.shape}")
        target = target[:, :, :, 0]  # 只保留第一个通道
    elif target.dim() == 4:  # [batch_size, 1, height, width]
        target = torch.squeeze(target, dim=1)
    
    # 确保输入和目标的空间维度匹配
    if input.shape[-2:] != target.shape[-2:]:
        print(f"调整输入尺寸从 {input.shape[-2:]} 到 {target.shape[-2:]}")
        # 使用目标的高度和宽度进行插值
        input = F.interpolate(input, size=(target.shape[-2], target.shape[-1]), 
                             mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

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
