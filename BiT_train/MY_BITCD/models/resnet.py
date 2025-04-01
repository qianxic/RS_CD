import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
"""
ResNet50模型实现文件
本文件实现了ResNet50深度卷积神经网络架构，专门用于变化检测任务中的特征提取
原始ResNet架构来自论文：'Deep Residual Learning for Image Recognition' (https://arxiv.org/pdf/1512.03385.pdf)
"""

__all__ = ['ResNet', 'resnet50']  # 只导出ResNet和resnet50函数


# 预训练模型URL
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',  # 官方预训练ResNet50模型权重
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3卷积层，ResNet的基本构建块
    
    参数:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 卷积步长
        groups: 分组卷积的组数
        dilation: 空洞卷积的膨胀率
    
    返回:
        3x3卷积层实例
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积层，用于改变通道数和降采样
    
    参数:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 卷积步长
    
    返回:
        1x1卷积层实例
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """Bottleneck块，ResNet50的基本构建块
    
    每个Bottleneck包含三个卷积层：
    - 1x1卷积降维
    - 3x3卷积特征提取
    - 1x1卷积升维
    还包含残差连接(shortcut connection)
    
    ResNet50中的结构是：
    输入 -> 1x1降维 -> 3x3卷积 -> 1x1升维 -> 输出
      |                                      |
      ----------------> shortcut ------------|
      
    """
    
    # Bottleneck模块中的扩展系数，表示输出通道数是中间层的4倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """初始化Bottleneck块
        
        参数:
            inplanes: 输入通道数
            planes: 中间层通道数(输出通道数将是planes*expansion)
            stride: 步长，用于降采样
            downsample: 下采样层，用于残差连接
            groups: 分组卷积的组数
            base_width: 基础宽度
            dilation: 空洞卷积的膨胀率
            norm_layer: 标准化层类型
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # 第一个1x1卷积层，降维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # 第二个3x3卷积层，特征提取，可能包含下采样（stride>1）
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        # 第三个1x1卷积层，升维
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 残差连接的下采样层
        self.stride = stride

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            处理后的特征图
        """
        identity = x  # 保存输入用于残差连接

        # 第一个1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个3x3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)#计算出来了结束的特征图

        out += identity  # 加上残差连接  y=x+f(x)
        out = self.relu(out)  # 最后的ReLU激活

        return out


class ResNet(nn.Module):
    """ResNet网络架构
    
    包含5个阶段：
    1. 初始卷积和池化
    2. layer1 (conv2_x): 包含3个Bottleneck块
    3. layer2 (conv3_x): 包含4个Bottleneck块
    4. layer3 (conv4_x): 包含6个Bottleneck块
    5. layer4 (conv5_x): 包含3个Bottleneck块
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        """初始化ResNet网络
        
        参数:
            block: 基本构建块类型 (Bottleneck)
            layers: 每个阶段的块数量列表，例如[3,4,6,3]
            num_classes: 分类任务的类别数
            zero_init_residual: 是否将残差块的最后BN层初始化为0
            groups: 分组卷积的组数
            width_per_group: 每组的宽度
            replace_stride_with_dilation: 是否用空洞卷积替换下采样
            norm_layer: 标准化层类型
            strides: 每个阶段的步长
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 设置每个阶段的步长
        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64  # 初始通道数
        self.dilation = 1
        
        # 空洞卷积配置
        if replace_stride_with_dilation is None:
            # 每个元素表示是否使用空洞卷积代替2x2步长
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # 第一个卷积层 - 7x7卷积，步长为2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        
        # 四个阶段的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])  # 不下采样
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                       dilate=replace_stride_with_dilation[0])  # 下采样
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                       dilate=replace_stride_with_dilation[1])  # 下采样
        self.layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                       dilate=replace_stride_with_dilation[2])  # 下采样
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化每个残差块中的最后BN层
        # 这样残差块一开始就类似于恒等映射，有助于模型训练
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """创建ResNet的一个阶段，包含多个块
        
        参数:
            block: 块类型
            planes: 基础通道数
            blocks: 块的数量
            stride: 是否对第一个块进行下采样
            dilate: 是否使用空洞卷积
            
        返回:
            包含多个块的Sequential模块
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 空洞卷积设置
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # 如果需要下采样或通道数不匹配，创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 添加第一个块，可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        # 添加剩余的块
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """实际的前向传播逻辑
        
        参数:
            x: 输入图像
            
        返回:
            分类结果
        """
        # 第一阶段: 卷积+池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        """前向传播接口函数
        
        参数:
            x: 输入图像
            
        返回:
            分类结果
        """
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    """创建ResNet模型
    
    参数:
        arch: 模型名称
        block: 块类型
        layers: 每个阶段的块数量
        pretrained: 是否加载预训练权重
        progress: 是否显示进度条
        **kwargs: 其他参数
        
    返回:
        ResNet模型实例
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """创建ResNet-50模型
    
    ResNet-50配置 [3, 4, 6, 3] 表示四个阶段分别有3, 4, 6, 3个Bottleneck块
    
    参数:
        pretrained: 是否加载预训练权重
        progress: 是否显示进度条
        **kwargs: 其他参数
        
    返回:
        ResNet-50模型实例
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
