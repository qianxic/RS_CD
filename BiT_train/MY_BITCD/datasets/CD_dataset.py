"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np

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

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        
        # 打印数据集信息
        print(f"[数据集信息] 模式: {split}, 图像总数: {self.A_size}")
        
        # 检查前5个图像的标签文件是否存在
        for i in range(min(5, self.A_size)):
            name = self.img_name_list[i]
            label_path = get_label_path(self.root_dir, name)
            print(f"[数据集检查] 样本 {i}: {name}, 标签文件存在: {os.path.exists(label_path)}")
            
        # 采样几个图像检查标签分布
        if self.A_size > 0:
            sample_indices = np.linspace(0, self.A_size-1, min(5, self.A_size)).astype(int)
            for i in sample_indices:
                name = self.img_name_list[i]
                label_path = get_label_path(self.root_dir, name)
                if os.path.exists(label_path):
                    try:
                        label_img = Image.open(label_path)
                        if label_img.mode != 'L':
                            label_img = label_img.convert('L')
                        label = np.array(label_img.resize((256, 256), Image.NEAREST), dtype=np.uint8)
                        if self.label_transform == 'norm':
                            label = label // 255
                        change_ratio = np.sum(label == 1) / label.size
                        print(f"[数据集检查] 样本 {i} ({name}) 变化区域占比: {change_ratio:.6f}, 标签唯一值: {np.unique(label)}")
                    except Exception as e:
                        print(f"[数据集检查] 样本 {i} ({name}) 标签读取错误: {str(e)}")

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        # 读取并调整图像尺寸为256x256
        img_A = np.asarray(Image.open(A_path).convert('RGB').resize((256, 256), Image.BILINEAR))
        img_B = np.asarray(Image.open(B_path).convert('RGB').resize((256, 256), Image.BILINEAR))
        
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        # 简化标签处理：直接转为单通道二值图像
        try:
            label_img = Image.open(L_path)
            # 强制转换为灰度图，确保单通道
            if label_img.mode != 'L':
                label_img = label_img.convert('L')
                
            # 调整标签尺寸为256x256，使用NEAREST方法保持二值特性
            label_img = label_img.resize((256, 256), Image.NEAREST)
            
            # 转为numpy数组，确保是二维的
            label = np.array(label_img, dtype=np.uint8)
            
            # 打印标签原始值统计
            if index % 100 == 0:  # 只打印部分数据
                unique_vals = np.unique(label)
                print(f"标签加载 [{name}] 原始唯一值: {unique_vals}, 形状: {label.shape}")
            
            # 二分类标签处理（如果需要）
            if self.label_transform == 'norm':
                label = label // 255
                
                # 打印归一化后的标签统计
                if index % 100 == 0:
                    norm_unique = np.unique(label)
                    change_ratio = np.sum(label == 1) / label.size
                    print(f"标签归一化后 [{name}] 唯一值: {norm_unique}, 变化比例: {change_ratio:.6f}")
                
        except Exception as e:
            print(f"标签处理错误 {L_path}: {e}")
            # 创建空白标签
            label = np.zeros((256, 256), dtype=np.uint8)

        # 转换为张量
        [img, img_B], [label] = self.augm.transform([img_A, img_B], [label], to_tensor=self.to_tensor)
        
        return {'name': name, 'A': img, 'B': img_B, 'L': label}

