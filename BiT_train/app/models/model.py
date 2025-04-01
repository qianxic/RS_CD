import torch
import torch.nn as nn
import os
import sys

# 添加MY_BITCD目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MY_BITCD'))

# 导入BIT模型
from MY_BITCD.models.networks import BIT

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
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        try:
            # 初始化BIT模型
            self.model = BIT(backbone='resnet18', output_stride=16, class_num=2)
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 这里根据模型的保存方式选择加载方法
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # 直接加载整个模型
                self.model = checkpoint
                
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            print("模型加载成功")
            
        except Exception as e:
            raise Exception(f"加载模型时出错: {str(e)}")
    
    def predict(self, input_data):
        """
        使用模型进行预测
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            模型预测结果
        """
        if self.model is None:
            raise Exception("模型未加载，请先调用load_model方法")
        
        # 确保输入数据在正确的设备上
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        
        # 执行前向传播
        with torch.no_grad():
            output = self.model(input_data)
            
        return output 