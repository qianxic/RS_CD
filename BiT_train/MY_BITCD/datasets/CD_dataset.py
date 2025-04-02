"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import torch

from torch.utils import data

from datasets.data_utils import CDDataAugmentation


"""
CD data set with pixel-level labels；
Modified structure:
/LEVIR-CD
    /train
        /A - 前时相图像
        /B - 后时相图像
        /label - 标签
    /val 
        /A
        /B
        /label
    /test
        /A
        /B
        /label
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.jpg' # jpg for gan dataset, others : png

def get_img_paths_from_dir(root_dir, folder_name):
    """获取指定目录下所有图像的路径和名称"""
    img_dir = os.path.join(root_dir, folder_name)
    img_list = []
    img_names = []
    
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(('.jpg', '.png')):
            img_list.append(os.path.join(img_dir, filename))
            img_names.append(filename)
    
    return img_list, img_names


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    # 尝试直接使用原始文件名（可能是.jpg）
    label_path = os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name)
    if os.path.exists(label_path):
        return label_path
        
    # 如果不存在，尝试替换.jpg为.png
    png_label = os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', '.png'))
    if os.path.exists(png_label):
        return png_label
        
    # 如果仍不存在，返回原始的.jpg路径（由调用方处理错误）
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name)


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = os.path.join(root_dir, split)  # 修改为直接使用train/val/test子目录
        self.img_size = img_size
        self.split = split  # train | val | test
        
        # 直接从目录中获取图像路径和名称
        _, self.img_name_list = get_img_paths_from_dir(self.root_dir, IMG_FOLDER_NAME)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, split='train', img_size=256, is_train=True,
                 label_transform='norm'):
        """
        初始化变化检测数据集
        
        参数:
            root_dir: 数据集根目录
            split: 数据集划分，如'train', 'val', 'test'
            img_size: 图像尺寸
            is_train: 是否为训练模式
            label_transform: 标签转换方式
        """
        # 记录参数
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.is_train = is_train
        self.label_transform = label_transform
        self.to_tensor = True
        
        # 构建数据目录路径 - 修正路径结构
        self.img1_dir = os.path.join(root_dir, 'A', split)
        self.img2_dir = os.path.join(root_dir, 'B', split)
        self.lab_dir = os.path.join(root_dir, 'label', split)
        
        # 如果目录不存在，尝试使用不同的目录结构
        if not os.path.exists(self.img1_dir):
            # 尝试直接使用split子目录
            self.img1_dir = os.path.join(root_dir, split, 'A')
            self.img2_dir = os.path.join(root_dir, split, 'B')
            self.lab_dir = os.path.join(root_dir, split, 'label')
            
        # 获取图像文件名列表
        self.img_name_list = self.get_ids(self.img1_dir)
        
        # 确保有图像
        if len(self.img_name_list) == 0:
            raise RuntimeError(f"找不到图像! 检查路径: {self.img1_dir}")
            
        # 记录数据集大小
        self.A_size = len(self.img_name_list)
       
        # 不再使用增强，仅创建基本的转换对象
        self.augm = CDDataAugmentation(
            img_size=self.img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_scale_random_crop=False,
            with_random_blur=False,
        )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        
        # 根据目录结构构建路径
        A_path = os.path.join(self.img1_dir, name)
        B_path = os.path.join(self.img2_dir, name)
        L_path = os.path.join(self.lab_dir, name)
        
        # 读取并调整图像尺寸为self.img_size
        img_A = np.asarray(Image.open(A_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR))
        img_B = np.asarray(Image.open(B_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR))

        # 简化标签处理：直接转为单通道二值图像
        try:
            label_img = Image.open(L_path)
            # 强制转换为灰度图，确保单通道
            if label_img.mode != 'L':
                label_img = label_img.convert('L')
                
            # 调整标签尺寸，使用NEAREST方法保持二值特性
            label_img = label_img.resize((self.img_size, self.img_size), Image.NEAREST)
            
            # 转为numpy数组，确保是二维的
            label = np.array(label_img, dtype=np.uint8)
            
            # 二分类标签处理（如果需要）
            if self.label_transform == 'norm':
                # 将所有非零值归一化为1
                label = np.where(label > 0, 1, 0).astype(np.uint8)
            else:
                # 无论如何都确保标签只有0和1
                label = np.where(label > 0, 1, 0).astype(np.uint8)
                
        except Exception as e:
            print(f"标签处理错误 {L_path}: {e}")
            # 创建空白标签
            label = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 转换为张量
        [img, img_B], [label] = self.augm.transform([img_A, img_B], [label], to_tensor=self.to_tensor)
        
        # 最后验证标签值
        if self.to_tensor and torch.max(label) > 1:
            print(f"警告：标签中包含大于1的值 [{name}]，最大值: {torch.max(label).item()}")
            # 修复标签值
            label = torch.clamp(label, 0, 1)
            
        return {'name': name, 'A': img, 'B': img_B, 'L': label}

    def get_ids(self, dir_path):
        """
        获取目录中的文件名列表
        
        参数:
            dir_path: 目录路径
            
        返回:
            文件名列表
        """
        return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

