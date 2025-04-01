from argparse import ArgumentParser
import torch
from models.trainer import *
import utils
import os

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    # 确保模型输出尺寸为256x256
    print(f"测试阶段输出尺寸: {args.img_size}x{args.img_size}")
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    # GPU配置参数
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # 项目名称，用于创建和识别不同实验的文件夹
    parser.add_argument('--project_name', default='test', type=str)
    # 检查点保存的根目录
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # 数据加载参数
    parser.add_argument('--num_workers', default=2, type=int)  # 数据加载的工作线程数，减少以降低内存使用
    parser.add_argument('--dataset', default='CDDataset', type=str)  # 使用的数据集类名称
    parser.add_argument('--data_name', default='LEVIR', type=str)  # 具体的数据集名称，修改为LEVIR

    # 训练参数
    parser.add_argument('--batch_size', default=8, type=int)  # 批处理大小，减小以防止内存溢出
    parser.add_argument('--split', default="train", type=str)  # 训练集划分
    parser.add_argument('--split_val', default="val", type=str)  # 验证集划分

    # 图像预处理参数
    parser.add_argument('--img_size', default=256, type=int)  # 输入图像大小

    # 模型参数
    parser.add_argument('--n_class', default=2, type=int)  # 分类类别数（变化/未变化）
    # 生成器网络类型，决定使用哪种模型架构
    parser.add_argument('--net_G', default='vit_base_patch16', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8 | vit_base_patch16')
    # 损失函数类型
    parser.add_argument('--loss', default='ce', type=str)

    # 优化器参数
    parser.add_argument('--optimizer', default='adam', type=str)  # 优化器类型，改为adam更稳定
    parser.add_argument('--lr', default=0.001, type=float)  # 学习率增大到0.001
    parser.add_argument('--max_epochs', default=100, type=int)  # 最大训练轮数
    # 学习率调整策略
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)  # 学习率衰减间隔

    # 解析命令行参数
    args = parser.parse_args()
    
    # 强制使用256x256尺寸
    args.img_size = 256
    
    # 对于VIT模型，确保图像尺寸合适
    if args.net_G == 'vit_base_patch16' and args.img_size != 224:
            pass
    
    # 设置设备（GPU/CPU）
    utils.get_device(args)
    print(args.gpu_ids)

    # 创建检查点目录
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # 创建可视化结果目录
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # 为调试添加CUDA阻塞模式
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 调用训练函数
    train(args)

    # 调用测试函数
    test(args)
