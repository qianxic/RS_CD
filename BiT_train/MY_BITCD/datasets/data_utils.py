import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    # 确保标签是2D张量(一个通道，无RGB)
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                        for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        # 增强的随机裁剪 - 智能变化区域裁剪
        if self.with_random_crop and random.random() > 0 and len(labels) > 0:
            # 将标签转换为numpy数组以便分析
            label_array = np.array(labels[0])
            
            # 检查标签是否有变化区域(值为255或1的像素)
            if label_array.max() > 0:
                # 查找变化区域的坐标
                if label_array.max() > 1:  # 如果标签是255值
                    change_y, change_x = np.where(label_array > 128)
                else:  # 如果标签是二值(0/1)
                    change_y, change_x = np.where(label_array > 0)
                
                # 只有找到变化区域时才进行智能裁剪
                if len(change_y) > 0:
                    # 计算变化区域的边界框
                    min_y, max_y = change_y.min(), change_y.max()
                    min_x, max_x = change_x.min(), change_x.max()
                    
                    # 设置裁剪区域大小 (确保至少包含80%的变化区域)
                    crop_size = min(self.img_size, max(max_y - min_y, max_x - min_x) * 5 // 4)
                    
                    # 随机选择中心点 (确保中心点在变化区域内或附近)
                    center_y = min_y + random.randint(0, max(1, max_y - min_y))
                    center_x = min_x + random.randint(0, max(1, max_x - min_x))
                    
                    # 计算裁剪框 (确保在图像范围内)
                    i = max(0, center_y - crop_size // 2)
                    j = max(0, center_x - crop_size // 2)
                    
                    # 调整裁剪框以确保不超出图像边界
                    if i + crop_size > label_array.shape[0]:
                        i = label_array.shape[0] - crop_size
                    if j + crop_size > label_array.shape[1]:
                        j = label_array.shape[1] - crop_size
                        
                    # 确保i和j为非负数
                    i, j = max(0, i), max(0, j)
                    
                    # 应用裁剪
                    imgs = [TF.resized_crop(img, i, j, crop_size, crop_size,
                                        size=(self.img_size, self.img_size),
                                        interpolation=Image.BICUBIC)
                        for img in imgs]

                    labels = [TF.resized_crop(img, i, j, crop_size, crop_size,
                                          size=(self.img_size, self.img_size),
                                          interpolation=Image.NEAREST)
                          for img in labels]
                else:
                    # 当没有找到变化区域时，退回到普通随机裁剪
                    i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                        get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(0.8, 1.2))
                        
                    imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.BICUBIC)
                        for img in imgs]

                    labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                          for img in labels]
            else:
                # 当标签全为0时，退回到普通随机裁剪
                i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                    get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(0.8, 1.2))
                    
                imgs = [TF.resized_crop(img, i, j, h, w,
                                size=(self.img_size, self.img_size),
                                interpolation=Image.BICUBIC)
                    for img in imgs]

                labels = [TF.resized_crop(img, i, j, h, w,
                                  size=(self.img_size, self.img_size),
                                  interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            
            # 简化标签处理：确保是单通道标签
            for i in range(len(labels)):
                if isinstance(labels[i], np.ndarray):
                    # 如果标签是numpy数组，确保是2D的
                    if labels[i].ndim > 2:
                        # 静默处理，不再输出警告
                        labels[i] = labels[i][:, :, 0] if labels[i].ndim == 3 else labels[i]

            # 转为张量，添加通道维度（确保是[1,H,W]形式）
            processed_labels = []
            for img in labels:
                # 转换为numpy数组（如果不是）并确保是uint8类型
                img_array = np.array(img, np.uint8)
                # 添加通道维度，确保形状是[1,H,W]
                label_tensor = torch.from_numpy(img_array).unsqueeze(dim=0)
                processed_labels.append(label_tensor)
            
            labels = processed_labels

            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
