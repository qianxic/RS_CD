# MY_BITCD - 变化检测模型架构

本目录包含了基于BIT（双时态变换器）的变化检测模型的核心实现，移植自原始BIT_CD项目。以下是各个文件和目录的功能介绍。

## 目录结构

```
MY_BITCD/
├── datasets/          # 数据集加载和处理代码
├── models/            # 模型架构的核心实现
├── data_config.py     # 数据配置文件
├── eval_cd.py         # 模型评估脚本
├── imutils.py         # 图像处理工具
├── main_cd.py         # 主训练脚本
├── metric_tool.py     # 评估指标工具
└── utils.py           # 通用工具函数
```

## 核心文件功能

### models/目录

模型架构的核心实现，包含以下文件：

- **networks.py**: 
  - 定义BIT（双时态变换器）网络架构
  - 这是整个变化检测模型的核心
  - 实现了基于Transformer的变化检测算法

- **resnet.py**: 
  - 实现ResNet骨干网络
  - 用于从输入图像中提取特征
  - 支持不同深度的ResNet变体（如ResNet18、ResNet50等）

- **help_funcs.py**: 
  - 包含各种辅助函数和网络组件
  - 实现注意力机制
  - 包含Transformer编码器和解码器
  - 各种网络层和模块的实现

- **basic_model.py**: 
  - 定义基础模型类
  - 提供模型操作的通用接口
  - 处理模型的存储和加载

- **losses.py**: 
  - 实现各种损失函数
  - 包括交叉熵损失
  - 用于模型训练过程

- **trainer.py**: 
  - 实现模型训练流程和策略
  - 处理训练循环
  - 管理学习率调整和优化器

- **evaluator.py**: 
  - 提供模型评估功能
  - 计算评估指标
  - 生成可视化结果

- **__init__.py**: 
  - Python包初始化文件
  - 定义导入接口

### 其他核心文件

- **utils.py**: 
  - 通用工具函数集合
  - 数据加载和处理
  - 模型配置和初始化
  - 其他辅助功能

- **imutils.py**: 
  - 图像处理工具函数
  - 图像变换（缩放、裁剪、翻转等）
  - 数据增强操作
  - 图像可视化工具

- **metric_tool.py**: 
  - 评估指标工具
  - 精确度、召回率、F1分数等指标计算
  - IoU（交并比）计算
  - 其他变化检测相关的评估指标

- **main_cd.py**: 
  - 主训练脚本
  - 解析命令行参数
  - 配置训练环境
  - 启动模型训练和评估流程

- **eval_cd.py**: 
  - 评估脚本
  - 加载预训练模型
  - 在测试集上评估模型性能
  - 生成评估报告和可视化结果

- **data_config.py**: 
  - 数据配置文件
  - 定义数据集路径配置
  - 设置数据集特定参数
  - 指定标签转换策略

- **datasets/目录**: 
  - 数据加载和处理相关代码
  - 包含数据集类定义
  - 实现数据加载、预处理和增强

## 在FastAPI中使用

要在FastAPI应用中使用该模型，可以参考以下代码：

```python
import sys
import os

# 添加MY_BITCD目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MY_BITCD'))

# 导入BIT模型
from models.networks import BIT

class ChangeDetectionModel:
    def __init__(self):
        """
        初始化变化检测模型
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_model(self, model_path):
        """
        从指定路径加载模型
        """
        # 初始化BIT模型
        self.model = BIT(backbone='resnet18', output_stride=16, class_num=2)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 这里根据模型的保存方式选择加载方法
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, input_data):
        """
        使用模型进行预测
        """
        # 执行前向传播
        with torch.no_grad():
            output = self.model(input_data)
            
        return output
```

## 模型训练

使用以下命令可以训练模型：

```bash
python main_cd.py --img_size 256 --batch_size 8 --lr 0.01 --max_epochs 200 --net_G base_transformer_pos_s4_dd8 --gpu_ids 0 --data_name quick_start
```

## 模型评估

使用以下命令可以评估模型：

```bash
python eval_cd.py --img_size 256 --net_G base_transformer_pos_s4_dd8 --gpu_ids 0 --checkpoint_name YOUR_MODEL_PATH
``` 