import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import traceback
import time
import glob
from tqdm import tqdm

import utils
from models.networks import *

import torch
import torch.optim as optim
import torch.nn.functional as F

from models.losses import cross_entropy
import models.losses as losses

from misc.logger_tool import Logger, Timer

from utils import de_norm


class CDTrainer():
    """
    变化检测模型训练器
    负责模型的训练、评估和检查点管理
    """
    def __init__(self, args, dataloaders):
        """
        初始化训练器
        
        参数:
            args: 配置参数
            dataloaders: 包含训练和验证数据加载器的字典
        """
        self.dataloaders = dataloaders

        self.n_class = args.n_class        # 类别数量，变化检测通常为2（变化/未变化）
        # 定义生成器网络G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        # 设置计算设备（GPU或CPU）
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(f"使用设备: {self.device}")

        # 学习率设置
        self.lr = args.lr
        
        # 边缘损失权重
        self.edge_weight = args.edge_weight if hasattr(args, 'edge_weight') else 0.3
        print(f"边缘损失权重: {self.edge_weight}")

        # 定义优化器
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr, 
                                     weight_decay=5e-4)

        # 定义学习率调度器
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        # 初始化评估指标计算器 - 使用SimpleMetrics替代ConfuseMatrixMeter
        self.running_metric = SimpleMetrics()

        # 设置检查点和可视化目录
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        
        # 确保目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 设置日志文件
        logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        
        # 初始化计时器
        self.timer = Timer()
        self.batch_size = args.batch_size

        # 初始化训练日志相关变量
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        
        # 早停设置
        self.patience = 20  # 20轮没有提升则停止
        self.early_stop_counter = 0
        self.should_early_stop = False
        
        # 周期性保存设置
        self.save_freq = 10  # 每10轮保存一次

        # 训练步数计算
        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        # 存储中间结果的变量
        self.G_pred = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        
        # 配置类别权重处理类别不平衡问题 - 为变化区域分配更高的权重
        self.class_weights = torch.tensor([1.0, 100.0], device=self.device)  
        print(f"类别权重: {self.class_weights}, 设备: {self.class_weights.device}")
        
        # 添加焦点损失参数
        self.use_focal_loss = True  # 开启焦点损失
        self.focal_gamma = 2.0  # 聚焦参数，增加对难分类样本的关注
        self.focal_alpha = 0.25  # 控制类别平衡的alpha参数
        print(f"使用焦点损失: gamma={self.focal_gamma}, alpha={self.focal_alpha}")
        
        # 设置图像大小
        self.img_size = args.img_size if hasattr(args, 'img_size') else 256
        
        # 记录模型信息
        print(f"\n模型信息:")
        print(f"使用网络: {args.net_G}")
        print(f"图像大小: {self.img_size}x{self.img_size}")
        print(f"批大小: {self.batch_size}")
        if args.net_G == 'vit_base_patch16':
            print("使用ViT作为特征提取器")
            print(f"嵌入维度: {args.embed_dim}")
            print(f"深度: {args.depth}")
            print(f"注意力头数: {args.num_heads}")
            
        # 尝试加载最新检查点
        self._load_checkpoint()

    def _load_checkpoint(self, checkpoint_path=None):
        """
        加载检查点
        
        args:
            checkpoint_path: 检查点路径，如果为None则尝试加载最新检查点
        """
        # 如果未指定路径，尝试加载'last_ckpt.pth'
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'last_ckpt.pth')
        
        # 如果'last_ckpt.pth'不存在，尝试遍历目录加载最新的检查点
        if not os.path.exists(checkpoint_path):
            # 获取所有epoch_*_ckpt.pth文件
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'epoch_*_ckpt.pth'))
            if checkpoint_files:
                # 按照修改时间排序，获取最新的检查点
                checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
        
        # 如果找到了检查点，加载它
        if os.path.exists(checkpoint_path):
            print(f"正在加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型权重
            if 'net_G' in checkpoint:
                self.net_G.load_state_dict(checkpoint['net_G'])
            elif 'model_G_state_dict' in checkpoint:
                self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            
            # 加载优化器状态
            if 'optimizer_G' in checkpoint:
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            elif 'optimizer_G_state_dict' in checkpoint:
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            
            # 加载学习率调度器状态(如果有)
            if hasattr(self, 'lr_scheduler_G') and self.lr_scheduler_G is not None:
                if 'lr_scheduler_G' in checkpoint:
                    self.lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
                elif 'exp_lr_scheduler_G_state_dict' in checkpoint:
                    self.lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            
            # 加载训练状态
            self.epoch_to_start = checkpoint.get('epoch', 0) + 1
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_epoch_id = checkpoint.get('best_epoch_id', 0)
            self.early_stop_counter = checkpoint.get('early_stop_counter', 0)
            
            print(f"成功加载检查点! 将从第 {self.epoch_to_start} 轮开始训练")
            print(f"历史最佳性能: Epoch {self.best_epoch_id}, mF1={self.best_val_acc:.4f}")
            
            return True
        else:
            print("未找到检查点，将从头开始训练")
            return False

    def _timer_update(self):
        """
        更新计时器并返回训练速度和预计剩余时间
        
        返回:
            imps: 每秒处理的图像数
            est: 预计剩余小时数
        """
        # 计算全局步数
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        # 更新进度
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()  # 预计剩余时间（小时）
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()  # 每秒处理的样本数
        return imps, est

    def _forward_pass(self, batch):
        """
        执行前向传播
        
        参数:
            batch: 批次数据
        """
        # 保存批次数据
        self.batch = batch
        
        # 获取输入张量
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        # 前向传播
        self.G_pred = self.net_G(img_in1, img_in2)
        
        # 在验证阶段打印尺寸信息(仅在第一个批次)
        if not self.is_training and self.batch_id == 0 and self.epoch_id == 0:
            print(f"\n网络维度信息:")
            print(f"输入A尺寸: {img_in1.shape}")
            print(f"输入B尺寸: {img_in2.shape}")
            
            if isinstance(self.G_pred, dict):
                print(f"模型输出格式: 字典格式")
                print(f"分割输出尺寸: {self.G_pred['seg'].shape}")
                print(f"边缘输出尺寸: {self.G_pred['edge'].shape}")
            else:
                print(f"模型输出尺寸: {self.G_pred.shape}")
                
            print(f"标签尺寸: {batch['L'].shape}\n")
            
    def _update_metric(self, pred):
        """
        更新评估指标
        
        参数:
            pred: 模型的预测结果
        """
        # 获取标签
        label = self.batch['L'].to(self.device)
        if label.dim() > 3:
            label = label.squeeze(1)
            
        # 如果预测是字典格式，使用分割结果更新指标
        if isinstance(pred, dict):
            prediction = torch.argmax(pred['seg'], dim=1)
        else:
            prediction = torch.argmax(pred, dim=1)
            
        # 更新评估指标
        self.running_metric.update(prediction.cpu().detach().numpy(), label.cpu().detach().numpy())

    def _visualize_pred(self):
        """
        生成预测结果的可视化
        
        返回值:
            pred_vis: 预测结果可视化 [B, 3, H, W]
        """
        # 获取边缘检测图
        edge_exists = False
        
        # 处理预测结果
        if isinstance(self.G_pred, dict):
            # 获取分割预测
            pred = torch.argmax(self.G_pred['seg'], dim=1, keepdim=True)
            # 如果存在边缘检测结果
            if 'edge' in self.G_pred:
                edge_exists = True
                edge_pred = (torch.sigmoid(self.G_pred['edge']) > 0.5).float()
        else:
            # 直接获取分割预测
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        
        # 创建可视化图像
        pred_vis = torch.zeros(pred.size(0), 3, pred.size(2), pred.size(3), device=pred.device)
        
        # 设置颜色：
        # 变化区域为红色
        pred_vis[:, 0] = (pred == 1).float().squeeze(1)
        
        # 如果有边缘检测结果，在可视化中显示为蓝色
        if edge_exists:
            edge_pred = edge_pred.squeeze(1)
            # 保留原有的变化区域，并添加边缘
            pred_vis[:, 2] = edge_pred
        
        return pred_vis
        
    def _collect_running_batch_states(self):
        """
        收集当前批次的状态信息，包括损失和指标
        
        返回值:
            running_acc: 本批次准确率
        """
        # 获取评估指标
        current_metrics = self.running_metric.get_scores()
        
        # 提取准确率和F1分数
        running_acc = current_metrics['accuracy']
        running_f1 = current_metrics['f1']
        running_precision = current_metrics['precision'] 
        running_recall = current_metrics['recall']
        
        # 打印批次信息
        m_indicator = '' if self.is_training else '[验证]'
        
        # 获取当前损失，避免使用过时属性
        current_loss = 0.0
        if hasattr(self, 'train_losses_dict') and len(self.train_losses_dict) > 0:
            current_loss = self.train_losses_dict['total'] / max(1, self.batch_id + 1)
        
        # 打印指标
        if self.is_training:
            print(f"{m_indicator} E{self.epoch_id+1} B{self.batch_id} - 损失: {current_loss:.4f}, 准确率: {running_acc:.4f}, F1: {running_f1:.4f}, 精确率: {running_precision:.4f}, 召回率: {running_recall:.4f}")
        
        return running_acc

    def _generate_edge_gt(self, target):
        """
        从分割标签生成边缘检测的标签
        
        参数:
            target: 分割标签 [B, H, W]
            
        返回:
            edge_gt: 边缘标签 [B, 1, H, W]
        """
        # 将分割标签转换为one-hot编码
        # 首先获取批量大小和标签尺寸
        batch_size, height, width = target.size()
        
        # 创建边缘标签的空张量
        edge_gt = torch.zeros(batch_size, 1, height, width, device=target.device)
        
        # 对每个样本处理
        for b in range(batch_size):
            # 将标签转换为numpy数组以便使用opencv
            mask = target[b].cpu().numpy().astype(np.uint8)
            
            # 使用梯度计算边缘
            # 水平梯度
            h_gradient = np.abs(np.diff(mask, axis=0, prepend=0))
            # 垂直梯度
            v_gradient = np.abs(np.diff(mask, axis=1, prepend=0))
            
            # 合并梯度
            edge_map = np.maximum(h_gradient, v_gradient)
            
            # 边缘是梯度不为0的地方
            edge_map = (edge_map > 0).astype(np.float32)
            
            # 转回torch张量
            edge_gt[b, 0] = torch.from_numpy(edge_map).to(target.device)
        
        return edge_gt

    def _save_checkpoint(self, epoch, is_best=False, save_path=None):
        """
        保存模型检查点
        args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, f'last_ckpt.pth')
        
        state = {
            'epoch': epoch,
            'net_G': self.net_G.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'early_stop_counter': self.early_stop_counter,
        }
        
        # 添加学习率调度器状态
        if hasattr(self, 'lr_scheduler_G') and self.lr_scheduler_G is not None:
            state['lr_scheduler_G'] = self.lr_scheduler_G.state_dict()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        torch.save(state, save_path)
        
        # 如果是最佳模型，单独保存一份
        if is_best and 'best_ckpt' not in save_path:
            best_path = os.path.join(self.checkpoint_dir, 'best_ckpt.pth')
            torch.save(state, best_path)

    def _update_lr_schedulers(self):
        """
        更新学习率调度器，通常在每个epoch结束时调用
        """
        self.exp_lr_scheduler_G.step()

    def _collect_epoch_states(self, metric_manager, val=False):
        """
        收集并记录当前epoch的训练状态
        """
        # 获取指标
        score_dict = metric_manager.get_scores()
        
        # 准备指标消息
        phase = '验证' if val else '训练'
        avg_loss = self.validate_losses_dict['total'] if val else self.train_losses_dict['total']
        
        # 添加分割线使输出更清晰
        print('-' * 80)
        
        # 1. 总体指标
        print(f"[{phase}] 总体: 准确率={score_dict['accuracy']:.4f}, 精确率={score_dict['precision']:.4f}, 召回率={score_dict['recall']:.4f}, F1={score_dict['f1']:.4f}")
        
        # 2. 类别指标
        print(f"[{phase}] 类别0(未变化): F1={score_dict['f1_0']:.4f}, 精确率={score_dict['precision_0']:.4f}, 召回率={score_dict['recall_0']:.4f}")
        print(f"[{phase}] 类别1(变化): F1={score_dict['f1_1']:.4f}, 精确率={score_dict['precision_1']:.4f}, 召回率={score_dict['recall_1']:.4f}")
        
        # 3. IoU指标
        print(f"[{phase}] IoU: 类别0={score_dict['iou_0']:.4f}, 类别1={score_dict['iou_1']:.4f}, 平均={score_dict['mean_iou']:.4f}")
        
        # 4. 混淆矩阵和像素统计
        changed_pixels = score_dict['true_pos'] + score_dict['false_neg']
        unchanged_pixels = score_dict['true_neg'] + score_dict['false_pos']
        total_pixels = score_dict['total_pixels']
        
        print(f"[{phase}] 统计: TP={score_dict['true_pos']}, FP={score_dict['false_pos']}, FN={score_dict['false_neg']}, TN={score_dict['true_neg']}")
        print(f"[{phase}] 像素: 总数={total_pixels}, 变化={changed_pixels}({changed_pixels/total_pixels*100:.2f}%), 未变化={unchanged_pixels}({unchanged_pixels/total_pixels*100:.2f}%)")
        
        # 5. 损失信息
        print(f"[{phase}] 损失: 平均={avg_loss:.4f}")
        
        print('-' * 80)
        
        # 记录当前epoch的F1分数作为模型选择标准
        # 使用变化区域(类别1)的F1分数作为主要评价指标
        self.epoch_acc = score_dict['f1_1']
        
        # 构建完整消息用于日志记录
        complete_msg = f"[{phase}] E{self.epoch_id+1} - 总体F1={score_dict['f1']:.4f}, 类别1F1={score_dict['f1_1']:.4f}, 损失={avg_loss:.4f}"
        
        # 记录到日志文件
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.write(complete_msg)
        
        return score_dict

    def _update_checkpoints(self, current_epoch, train_score):
        """
        更新检查点
        """
        # 获取当前验证准确率
        current_acc = self.epoch_acc if hasattr(self, 'epoch_acc') else train_score.get('f1', 0)

        # 判断是否为最佳模型
        is_best = False
        
        # 如果当前准确率比历史最佳准确率高，则更新最佳准确率
        if current_acc > self.best_val_acc:
            is_best = True
            self.best_val_acc = current_acc
            self.best_epoch_id = current_epoch
            self.early_stop_counter = 0
            print(f"[更新] 第 {current_epoch+1} 轮: 新的最佳F1={current_acc:.4f}")
        else:
            # 增加早停计数器
            self.early_stop_counter += 1
            epochs_without_improvement = self.early_stop_counter
            print(f"[未改进] 第 {current_epoch+1} 轮: F1={current_acc:.4f}, 已连续 {epochs_without_improvement} 轮无改进 (最佳F1={self.best_val_acc:.4f})")
            
            # 如果超过早停耐心值，设置早停标志
            if self.early_stop_counter >= self.patience:
                self.should_early_stop = True
                print(f"[早停触发] 连续 {self.patience} 轮无改进")
        
        # 保存检查点
        if (current_epoch + 1) % self.save_freq == 0:
            filename = f'epoch_{current_epoch+1}_ckpt.pth'
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._save_checkpoint(current_epoch, is_best, save_path)
            print(f"[检查点] 已保存 {filename}")
        
        # 如果是最佳模型，保存为best_ckpt.pth
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_ckpt.pth')
            self._save_checkpoint(current_epoch, is_best, best_path)
            print(f"[检查点] 已保存最佳模型到 best_ckpt.pth")
        
        # 每10轮保存一次检查点(无论是否为最佳模型)
        if (current_epoch + 1) % 10 == 0:
            save_path = os.path.join(self.checkpoint_dir, f'epoch_{current_epoch+1}_ckpt.pth')
            self._save_checkpoint(current_epoch, is_best, save_path)
            print(f"[检查点] 已保存第 {current_epoch+1} 轮检查点")

    def _update_training_acc_curve(self):
        """
        更新训练准确率曲线记录
        """
        # 添加当前epoch的准确率到历史记录
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        # 保存更新后的历史记录
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        """
        更新验证准确率曲线记录
        """
        # 添加当前epoch的准确率到历史记录
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        # 保存更新后的历史记录
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        """
        清除缓存，主要是重置评估指标
        """
        self.running_metric.reset()

    def _pxl_loss(self, output, target, class_weights=None):
        """
        计算像素级损失，支持交叉熵和焦点损失
        
        参数:
            output: 模型输出，形状为[B, C, H, W]
            target: 目标标签，形状为[B, H, W]
            class_weights: 类别权重，形状为[C]
            
        返回:
            total_loss: 总损失
            ce_loss: 交叉熵损失
            focal_loss: 焦点损失(如果启用)
        """
        # 如果没有提供类别权重，使用默认值
        if class_weights is None:
            class_weights = self.class_weights
        
        # 确保类别权重在正确的设备上
        if class_weights.device != output.device:
            class_weights = class_weights.to(output.device)
        
        try:
            # 确保目标张量形状正确
            if target.dim() > 3:
                target = target.squeeze(1)  # 从[B,1,H,W]变为[B,H,W]
            
            # 确保目标张量类型为long
            if target.dtype != torch.long:
                target = target.long()
            
            # 将目标值限制在合法范围内
            target = torch.clamp(target, 0, self.n_class - 1)
            
            # 计算带权重的交叉熵损失
            ce_loss = F.cross_entropy(output, target, weight=class_weights, reduction='none')
            
            # 在计算整体损失和组件损失之前，先记录标量值
            ce_loss_scalar = ce_loss.mean()  # 标量值
            
            # 是否启用焦点损失
            if hasattr(self, 'use_focal_loss') and self.use_focal_loss:
                # 获取预测概率
                probs = F.softmax(output, dim=1)
                # 提取每个像素的目标类别概率
                pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
                # 计算焦点权重
                focal_weight = (1 - pt) ** self.focal_gamma
                # 应用焦点损失
                focal_loss = focal_weight * ce_loss
                # 计算平均焦点损失
                focal_loss_scalar = focal_loss.mean()  # 标量值
                # 总损失为带权重的焦点损失
                total_loss = ce_loss_scalar + focal_loss_scalar
                
                # 返回所有三个标量损失值
                return total_loss, ce_loss_scalar, focal_loss_scalar
            else:
                # 不使用焦点损失，总损失就是交叉熵损失
                return ce_loss_scalar
        
        except Exception as e:
            print(f"计算损失时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建一个虚拟损失返回，避免训练中断
            dummy_loss = torch.tensor(1.0, device=output.device, requires_grad=True)
            if hasattr(self, 'use_focal_loss') and self.use_focal_loss:
                return dummy_loss, dummy_loss, dummy_loss
            else:
                return dummy_loss

    def _save_visualization(self, suffix=''):
        """
        生成并保存训练过程的可视化图像
        """
        # 反归一化输入图像
        vis_input1 = utils.make_numpy_grid(de_norm(self.batch['A']))  # 第一时间点图像
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))  # 第二时间点图像

        # 获取预测结果可视化
        vis_pred = utils.make_numpy_grid(self._visualize_pred())

        # 处理真实标签进行可视化
        label_tensor = self.batch['L'].clone()
        
        # 确保标签维度正确，便于可视化
        if label_tensor.dim() > 4:
            label_tensor = label_tensor.squeeze(1)
        if label_tensor.dim() == 4 and label_tensor.size(1) > 1:
            label_tensor = label_tensor[:, 0:1, :, :]
        if label_tensor.dim() == 3:
            label_tensor = label_tensor.unsqueeze(1)

        # 标签可视化
        vis_gt = utils.make_numpy_grid(label_tensor * 255)  # 乘以255使得标签更清晰可见

        # 拼接基础可视化结果
        vis = np.concatenate([vis_input1, vis_input2, vis_gt, vis_pred], axis=0)
        
        vis = np.clip(vis, a_min=0.0, a_max=1.0) 
        
        # 根据训练/验证阶段命名
        phase = 'train' if self.is_training else 'val'
        
        # 创建文件名
        file_name = os.path.join(
            self.vis_dir, f'{phase}_{suffix}_e{self.epoch_id}_b{self.batch_id}.jpg')
        
        # 保存可视化图像
        plt.imsave(file_name, vis)

    def _load_best_before_continue(self, current_epoch):
        """
        在每10轮后，从最佳模型继续训练的函数
        
        args:
            current_epoch: 当前epoch
        
        返回值:
            bool: 是否成功加载了最佳模型
        """
        # 只在每10轮后执行
        if current_epoch % 10 != 0:
            return False
        
        # 检查是否有最佳模型
        best_ckpt_path = os.path.join(self.checkpoint_dir, 'best_ckpt.pth')
        if not os.path.exists(best_ckpt_path):
            print(f"第 {current_epoch} 轮: 未找到最佳模型，继续使用当前模型")
            return False
        
        # 如果最佳模型是当前模型，不需要加载
        if self.best_epoch_id == current_epoch - 1:
            print(f"第 {current_epoch} 轮: 当前模型已经是最佳模型，继续训练")
            return False
        
        # 加载最佳模型
        print(f"第 {current_epoch} 轮: 加载第 {self.best_epoch_id} 轮的最佳模型继续训练")
        
        # 只加载模型权重，不加载其他训练状态
        checkpoint = torch.load(best_ckpt_path, map_location=self.device)
        if 'net_G' in checkpoint:
            self.net_G.load_state_dict(checkpoint['net_G'])
        elif 'model_G_state_dict' in checkpoint:
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        
        print(f"已加载最佳模型 (Epoch {self.best_epoch_id}, mF1: {self.best_val_acc:.4f})")
        return True

    def _validate(self, current_epoch, train_score):
        """执行验证阶段"""
        # 设置模型为评估模式
        self.is_training = False
        self.net_G.eval()
        
        # 重置指标管理器
        self.running_metric.reset()
        
        # 重置验证损失累计
        self.validate_losses_dict = {'total': 0.0, 'ce': 0.0, 'focal': 0.0}
        val_batch_count = 0
        
        print(f"\n[验证] 第 {current_epoch+1} 轮验证开始")
        
        # 遍历验证数据
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloaders['val'], desc=f"验证 Epoch {current_epoch+1}/{self.max_num_epochs}")):
                try:
                    # 获取批次数据并转移到设备
                    A_img = batch['A'].to(self.device)
                    B_img = batch['B'].to(self.device)
                    target = batch['L'].to(self.device)
                    
                    # 前向传播
                    output = self.net_G(A_img, B_img)
                    
                    # 计算损失
                    losses = self._pxl_loss(output, target, self.class_weights)
                    if isinstance(losses, tuple) and len(losses) == 3:
                        loss_total, loss_ce, loss_focal = losses
                        
                        # 累计损失（转为标量）
                        current_total = loss_total.item() if hasattr(loss_total, 'item') else float(loss_total)
                        current_ce = loss_ce.item() if hasattr(loss_ce, 'item') else float(loss_ce)
                        current_focal = loss_focal.item() if hasattr(loss_focal, 'item') else float(loss_focal)
                        
                        self.validate_losses_dict['total'] += current_total
                        self.validate_losses_dict['ce'] += current_ce
                        self.validate_losses_dict['focal'] += current_focal
                    else:
                        # 单一损失情况
                        loss_total = losses
                        
                        # 累计损失（转为标量）
                        current_total = loss_total.item() if hasattr(loss_total, 'item') else float(loss_total)
                        self.validate_losses_dict['total'] += current_total
                    
                    # 更新指标
                    if isinstance(output, tuple):
                        output = output[0]  # 取第一个元素作为预测结果
                    
                    # 确保输出和目标具有相同的形状
                    if output.shape != target.shape:
                        output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                    
                    # 获取预测类别
                    prediction = torch.argmax(output, dim=1)
                    self.running_metric.update(prediction.cpu().detach().numpy(), target.cpu().detach().numpy())
                    
                    # 增加批次计数
                    val_batch_count += 1
                    
                    # 每隔一段时间打印当前批次信息
                    if i % 20 == 0 or i == len(self.dataloaders['val']) - 1:
                        # 计算当前平均损失
                        avg_total = self.validate_losses_dict['total'] / max(1, val_batch_count)
                        avg_ce = self.validate_losses_dict['ce'] / max(1, val_batch_count)
                        avg_focal = self.validate_losses_dict['focal'] / max(1, val_batch_count)
                        
                        print(f"验证批次 {i+1}/{len(self.dataloaders['val'])}, 损失: {avg_total:.4f}")
                
                except Exception as e:
                    print(f"[验证] 在批次 {i} 处理时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 计算验证阶段的平均损失
        if val_batch_count > 0:
            self.validate_losses_dict['total'] /= val_batch_count
            self.validate_losses_dict['ce'] /= val_batch_count
            self.validate_losses_dict['focal'] /= val_batch_count
        
        # 收集验证指标
        print(f"\n[验证] 第 {current_epoch+1} 轮验证结束, 共 {val_batch_count} 批次")
        
        # 获取指标
        val_score = self.running_metric.get_scores()
        
        # 打印验证信息
        phase = '验证'
        avg_loss = self.validate_losses_dict['total']
        metrics_msg = f"[{phase}] 准确率={val_score['accuracy']:.4f}, 精确率={val_score['precision']:.4f}, 召回率={val_score['recall']:.4f}, F1={val_score['f1']:.4f}"
        metrics_msg += f"\n[{phase}] TP={val_score['true_pos']}, FP={val_score['false_pos']}, FN={val_score['false_neg']}, TN={val_score['true_neg']}"
        metrics_msg += f"\n[{phase}] 平均损失={avg_loss:.4f}"
        print(metrics_msg)
        
        # 更新epoch_acc变量作为评估标准
        self.epoch_acc = val_score['f1']
        
        return val_score

    def train_models(self):
        """
        训练模型并返回训练完成的模型
        
        返回:
            net_G: 训练完成的模型
        """
        # 记录开始时间
        start_time = time.time()
        
        # 遍历指定轮数
        for current_epoch in range(self.epoch_to_start, self.max_num_epochs):
            # 设置为训练模式
            self.is_training = True
            self.net_G.train()
            self.epoch_id = current_epoch
            
            # 重置损失累计和评估指标
            self.train_losses_dict = {'total': 0.0, 'ce': 0.0, 'focal': 0.0}
            self.running_metric.reset()
            train_batch_count = 0
            
            print(f"[训练] {current_epoch+1}轮开始")
            
            # 训练循环
            for i, batch in enumerate(tqdm(self.dataloaders['train'], desc=f"训练 Epoch {current_epoch+1}/{self.max_num_epochs}")):
                # 记录批次ID
                self.batch_id = i
                
                try:
                    # 获取批次数据并转移到设备
                    A_img = batch['A'].to(self.device)
                    B_img = batch['B'].to(self.device)
                    target = batch['L'].to(self.device)
                    
                    # 确保标签维度正确
                    if target.dim() > 3:  # 如果标签是4D的 [B,1,H,W]
                        target = target.squeeze(1)  # 压缩到 [B,H,W]
                    
                    # 清空梯度
                    self.optimizer_G.zero_grad()
                    
                    # 前向传播
                    output = self.net_G(A_img, B_img)
                    self.G_pred = output  # 保存预测结果以便后续可视化
                    
                    # 计算损失
                    losses = self._pxl_loss(output, target, self.class_weights)
                    if isinstance(losses, tuple) and len(losses) == 3:
                        loss_total, loss_ce, loss_focal = losses
                        # 反向传播
                        loss_total.backward()
                        
                        # 累计损失（转为标量）
                        current_total = loss_total.item() if hasattr(loss_total, 'item') else float(loss_total)
                        current_ce = loss_ce.item() if hasattr(loss_ce, 'item') else float(loss_ce)
                        current_focal = loss_focal.item() if hasattr(loss_focal, 'item') else float(loss_focal)
                        
                        self.train_losses_dict['total'] += current_total
                        self.train_losses_dict['ce'] += current_ce
                        self.train_losses_dict['focal'] += current_focal
                    else:
                        # 单一损失情况
                        loss_total = losses
                        loss_total.backward()
                        
                        # 累计损失（转为标量）
                        current_total = loss_total.item() if hasattr(loss_total, 'item') else float(loss_total)
                        self.train_losses_dict['total'] += current_total
                    
                    # 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), 1.0)
                    
                    # 优化器更新
                    self.optimizer_G.step()
                    
                    # 更新指标
                    # 获取预测类别
                    if isinstance(output, dict):
                        prediction = torch.argmax(output['seg'], dim=1)
                    else:
                        prediction = torch.argmax(output, dim=1)
                    self.running_metric.update(prediction.cpu().numpy(), target.cpu().numpy())
                    
                    # 增加批次计数
                    train_batch_count += 1
                    
                    # 每隔一段时间打印当前批次信息
                    if i % 20 == 0 or i == len(self.dataloaders['train']) - 1:
                        # 计算当前平均损失
                        avg_total = self.train_losses_dict['total'] / max(1, train_batch_count)
                        avg_ce = self.train_losses_dict['ce'] / max(1, train_batch_count) if 'ce' in self.train_losses_dict else 0
                        avg_focal = self.train_losses_dict['focal'] / max(1, train_batch_count) if 'focal' in self.train_losses_dict else 0
                        
                        # 获取训练速度和预计剩余时间
                        speed, eta = self._timer_update()
                        print(f"[训练] E{current_epoch+1}/{self.max_num_epochs}, B{i}/{len(self.dataloaders['train'])-1}, Loss={avg_total:.4f}, CE={avg_ce:.4f}, Focal={avg_focal:.4f}, {speed:.1f}it/s, ETA:{eta:.1f}h")
                
                except Exception as e:
                    print(f"[训练] 批次 {i} 处理时出错: {e}")
                    traceback.print_exc()
                    continue
            
            # 计算训练阶段的平均损失
            if train_batch_count > 0:
                self.train_losses_dict['total'] /= train_batch_count
                if 'ce' in self.train_losses_dict:
                    self.train_losses_dict['ce'] /= train_batch_count
                if 'focal' in self.train_losses_dict:
                    self.train_losses_dict['focal'] /= train_batch_count
            
            # 打印训练阶段结束信息
            print(f"\n[训练] 第 {current_epoch+1} 轮结束, 共 {train_batch_count} 批次")
            train_score = self._collect_epoch_states(self.running_metric)
            
            # 如果有验证集，进行验证
            if len(self.dataloaders['val']) > 0:
                # 进行验证
                valid_score = self._validate(current_epoch, train_score)
            
            # 更新学习率
            self._update_lr_schedulers()
            
            # 更新检查点
            self._update_checkpoints(current_epoch, train_score)
            
            # 检查是否需要提前停止
            if self.should_early_stop:
                print(f"[提前停止] 在 {current_epoch+1} 轮训练后, 因为 {self.patience} 轮内没有改进")
                break
            
            # 每10轮后检查是否需要加载最佳模型继续训练
            if (current_epoch + 1) % 10 == 0 and current_epoch + 1 < self.max_num_epochs:
                self._load_best_before_continue(current_epoch + 1)
        
        # 记录训练结束时间
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
        # 保存最终模型
        self._save_checkpoint(epoch=self.max_num_epochs - 1, is_best=False, save_path=os.path.join(self.checkpoint_dir, f'final_ckpt.pth'))
        print(f"最终模型已保存到 {os.path.join(self.checkpoint_dir, 'final_ckpt.pth')}")
        
        # 如果有最佳模型，加载它
        best_checkpoint = os.path.join(self.checkpoint_dir, 'best_ckpt.pth')
        if os.path.exists(best_checkpoint):
            self._load_checkpoint(best_checkpoint)
            print(f"已加载最佳模型 (Epoch {self.best_epoch_id}, 验证F1: {self.best_val_acc:.4f})")
        
        return self.net_G

    def _print_metrics(self, scores, phase='Training', epoch=None):
        """
        打印评估指标
        
        参数:
            scores: 指标字典
            phase: 阶段名称('Training'或'Validation')
            epoch: 当前epoch
        """
        epoch_str = f"Epoch {epoch+1} " if epoch is not None else ""
        print(f"[{phase}] {epoch_str}结果:")
        print(f"  精确度: {scores['precision']:.4f}")
        print(f"  召回率: {scores['recall']:.4f}")
        print(f"  F1分数: {scores['f1']:.4f}")
        print(f"  准确率: {scores['accuracy']:.4f}")
        print(f"  TP: {scores['true_pos']}, FP: {scores['false_pos']}, TN: {scores['true_neg']}, FN: {scores['false_neg']}")
        
        # 记录到日志
        if hasattr(self, 'logger') and self.logger is not None:
            log_str = f"[{phase}] {epoch_str}- "
            log_str += f"精确度={scores['precision']:.4f}, "
            log_str += f"召回率={scores['recall']:.4f}, "
            log_str += f"F1={scores['f1']:.4f}, "
            log_str += f"准确率={scores['accuracy']:.4f}"
            self.logger.write(log_str)


class SimpleMetrics():
    """
    简单评估指标计算类，用于替代混淆矩阵
    """
    def __init__(self):
        """初始化指标计算器"""
        self.reset()
        
    def reset(self):
        """重置所有计数器"""
        self.tp = 0  # 真正例
        self.fp = 0  # 假正例
        self.tn = 0  # 真负例
        self.fn = 0  # 假负例
        self.total_pixels = 0  # 总像素数
        
    def update(self, pred, target):
        """
        更新指标计数
        
        参数:
            pred: 预测结果 numpy数组，形状为 [B, H, W]
            target: 目标标签 numpy数组，形状为 [B, H, W]
        """
        # 确保输入是numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
        if torch.is_tensor(target):
            target = target.cpu().detach().numpy()
            
        # 计算各类指标
        self.tp += np.sum((pred == 1) & (target == 1))  # 真正例：预测为1且实际为1
        self.fp += np.sum((pred == 1) & (target == 0))  # 假正例：预测为1但实际为0
        self.tn += np.sum((pred == 0) & (target == 0))  # 真负例：预测为0且实际为0
        self.fn += np.sum((pred == 0) & (target == 1))  # 假负例：预测为0但实际为1
        
        # 更新总像素数
        self.total_pixels += pred.size
        
    def get_scores(self):
        """
        计算并返回评估指标
        
        返回:
            dict: 包含准确率、精确率、召回率、F1分数等指标的字典
        """
        # 计算准确率
        accuracy = (self.tp + self.tn) / max(1, self.total_pixels)
        
        # 计算精确率 precision = TP / (TP + FP)
        precision = self.tp / max(1, self.tp + self.fp)
        
        # 计算召回率 recall = TP / (TP + FN)
        recall = self.tp / max(1, self.tp + self.fn)
        
        # 计算F1分数 F1 = 2 * precision * recall / (precision + recall)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        # 计算每个类的精确率和召回率
        # 类别0（未变化区域）
        precision_0 = self.tn / max(1, self.tn + self.fn)
        recall_0 = self.tn / max(1, self.tn + self.fp)
        f1_0 = 2 * precision_0 * recall_0 / max(1e-6, precision_0 + recall_0)
        
        # 类别1（变化区域）
        precision_1 = precision  # 这与上面计算的精确率相同
        recall_1 = recall  # 这与上面计算的召回率相同
        f1_1 = f1  # 这与上面计算的F1分数相同
        
        # 计算IoU
        iou_0 = self.tn / max(1, self.tn + self.fp + self.fn)  # 类别0 IoU
        iou_1 = self.tp / max(1, self.tp + self.fp + self.fn)  # 类别1 IoU
        mean_iou = (iou_0 + iou_1) / 2
        
        # 构建并返回包含所有指标的字典
        scores = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_0': precision_0,
            'recall_0': recall_0,
            'f1_0': f1_0,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'f1_1': f1_1,
            'iou_0': iou_0,
            'iou_1': iou_1,
            'mean_iou': mean_iou,
            'true_pos': int(self.tp),
            'false_pos': int(self.fp),
            'true_neg': int(self.tn),
            'false_neg': int(self.fn),
            'total_pixels': int(self.total_pixels),
            'changed_ratio': float(self.tp + self.fn) / max(1, self.total_pixels)
        }
        
        return scores

