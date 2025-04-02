import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):
    """获取训练和验证数据加载器"""
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    
    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    
    # 处理维度不兼容问题
    if tensor_data.dim() == 3 and tensor_data.size(0) == 1:
        # 单通道单个样本，形如 [1, H, W]
        # 复制通道到3通道
        tensor_data = tensor_data.repeat(3, 1, 1)
    elif tensor_data.dim() == 2:
        # 如果是2D张量 [H, W]，添加批次和通道维度
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
        # 复制到3通道
        tensor_data = tensor_data.repeat(1, 3, 1, 1)
    elif tensor_data.dim() == 3 and tensor_data.size(0) != 3 and tensor_data.size(0) != 1:
        # 批次张量，如 [B, H, W]，添加通道维度
        tensor_data = tensor_data.unsqueeze(1)
        # 复制到3通道
        tensor_data = tensor_data.repeat(1, 3, 1, 1)
    elif tensor_data.dim() == 4 and tensor_data.size(1) != 3 and tensor_data.size(1) != 1:
        # 错误形状的4D张量，如 [B, X, H, W]
        # 尝试将其重塑为 [B*X, 1, H, W]
        B, X, H, W = tensor_data.size()
        tensor_data = tensor_data.view(-1, 1, H, W)
        # 复制到3通道
        tensor_data = tensor_data.repeat(1, 3, 1, 1)
    elif tensor_data.dim() == 4 and tensor_data.size(1) == 1:
        # 单通道批次，形如 [B, 1, H, W]
        # 复制到3通道
        tensor_data = tensor_data.repeat(1, 3, 1, 1)
    
    try:
        vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
        vis = np.array(vis.cpu()).transpose((1,2,0))
        if vis.shape[2] == 1:
            vis = np.stack([vis, vis, vis], axis=-1)
        return vis
    except Exception as e:
        # 静默处理错误，返回替代图像
        # 紧急回退方案
        if tensor_data.dim() >= 2:
            # 取第一个样本或通道作为示例
            single_img = tensor_data.view(-1, *tensor_data.shape[-2:])
            single_img = single_img[0].unsqueeze(0)
            if single_img.dim() == 2:
                single_img = single_img.unsqueeze(0)
            # 将单通道复制为3通道
            if single_img.size(0) == 1:
                single_img = single_img.repeat(3, 1, 1)
            vis = np.array(single_img.cpu()).transpose((1,2,0))
            if vis.shape[2] == 1:
                vis = np.stack([vis, vis, vis], axis=-1)
            return vis
        else:
            # 生成一个空白图像
            return np.zeros((256, 256, 3))


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):#--gpu_ids字符串处理为设备ID列表：
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        # 兼容新版本PyTorch的设备设置方法
        try:
            torch.cuda.set_device(args.gpu_ids[0])
        except AttributeError:
            # 新版本PyTorch的替代方法
            device = torch.device(f'cuda:{args.gpu_ids[0]}')
            torch.cuda.device(device)