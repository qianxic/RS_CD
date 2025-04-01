import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import base64
import io

def process_images(image_before, image_after, target_size=(256, 256)):
    """
    处理输入的两张图像，进行预处理以适应BIT模型输入要求
    
    Args:
        image_before: 前时相图像 (PIL Image)
        image_after: 后时相图像 (PIL Image)
        target_size: 调整的目标大小
    
    Returns:
        预处理后可直接输入模型的张量字典
    """
    # 确保图像大小一致
    image_before = image_before.resize(target_size)
    image_after = image_after.resize(target_size)
    
    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 将图像转换为张量
    tensor_before = transform(image_before)
    tensor_after = transform(image_after)
    
    # 添加批次维度
    tensor_before = tensor_before.unsqueeze(0)
    tensor_after = tensor_after.unsqueeze(0)
    
    # BIT模型需要一个字典作为输入，包含A和B两个时间点的图像
    input_dict = {
        'A': tensor_before,
        'B': tensor_after
    }
    
    return input_dict

def decode_results(model_output):
    """
    解码模型输出结果
    
    Args:
        model_output: 模型的原始输出 (BIT模型通常输出logits)
    
    Returns:
        处理后的变化检测结果（变化图）
    """
    # BIT模型输出为字典格式，包含'change_pred'键
    if isinstance(model_output, dict) and 'change_pred' in model_output:
        change_pred = model_output['change_pred']
    else:
        change_pred = model_output
    
    # 如果输出是张量，转换为numpy数组
    if isinstance(change_pred, torch.Tensor):
        # 应用softmax并取最大值索引
        if change_pred.size(1) > 1:  # 多类别情况
            probs = torch.softmax(change_pred, dim=1)
            change_map = torch.argmax(probs, dim=1).cpu().numpy()
        else:  # 二分类情况
            change_map = (torch.sigmoid(change_pred) > 0.5).cpu().numpy().squeeze()
    else:
        change_map = change_pred
    
    # 确保输出是二维数组
    if len(change_map.shape) > 2:
        change_map = change_map.squeeze()
    
    # 将结果转换为8位图像
    binary_map = (change_map * 255).astype(np.uint8)
    
    return binary_map

def encode_image_to_base64(image_array):
    """
    将图像数组编码为base64字符串
    
    Args:
        image_array: 图像数组 (numpy array)
    
    Returns:
        base64编码的字符串
    """
    # 确保图像是8位的
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # 将numpy数组转换为PIL图像
    pil_image = Image.fromarray(image_array)
    
    # 将PIL图像编码为base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str 