import argparse
import torch
from models.trainer import CDTrainer
import utils
import os

def train(args):
    """
    训练变化检测模型
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    
    
    print("="*30)
    
    # 创建数据加载器
    dataloaders = utils.get_loaders(args)
    
    # 创建训练器并开始训练
    trainer = CDTrainer(args, dataloaders)
    trainer.train_models()

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    
    # 基本训练参数
    parser.add_argument('--project_name', type=str, default='test', 
                       help='项目名称，用于保存检查点和可视化结果')
    parser.add_argument('--checkpoint_root', type=str, default='checkpoints', 
                       help='检查点保存的根目录')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='数据加载的工作线程数')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='CDDataset', 
                       help='数据集类型')
    parser.add_argument('--data_name', type=str, default='LEVIR', 
                       help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='训练批大小')
    parser.add_argument('--split', type=str, default='train', 
                       help='训练集划分')
    parser.add_argument('--split_val', type=str, default='val', 
                       help='验证集划分')
    parser.add_argument('--img_size', type=int, default=512, 
                       help='输入图像尺寸')
    parser.add_argument('--n_class', type=int, default=2, 
                       help='类别数量')
    
    # 模型参数
    parser.add_argument('--net_G', type=str, default='vit_base_patch16', 
                       help='网络架构类型')
    parser.add_argument('--vit_img_size', type=int, default=448,
                       help='ViT模型接收的图像尺寸')
    parser.add_argument('--embed_dim', type=int, default=768, 
                       help='ViT嵌入维度')
    parser.add_argument('--depth', type=int, default=12, 
                       help='ViT深度')
    parser.add_argument('--num_heads', type=int, default=12, 
                       help='ViT注意力头数')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, 
                       help='ViT MLP层扩展比例')
    parser.add_argument('--attn_sparsity', type=float, default=0.1, 
                       help='注意力机制稀疏度')
    parser.add_argument('--fusion_sparsity', type=float, default=0.2, 
                       help='特征融合稀疏度')
    parser.add_argument('--edge_weight', type=float, default=0.5, 
                       help='边缘检测损失权重')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-5, 
                       help='初始学习率')
    parser.add_argument('--max_epochs', type=int, default=100, 
                       help='最大训练轮次')
    parser.add_argument('--lr_policy', type=str, default='linear', 
                       help='学习率调度策略: linear | step | plateau | cosine')
    
    # GPU参数
    parser.add_argument('--gpu_ids', type=str, default='0', 
                       help='使用的GPU ID，用逗号分隔')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置保存路径
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    args.vis_dir = os.path.join('vis', args.project_name)
    
    # 处理GPU IDs
    utils.get_device(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)
