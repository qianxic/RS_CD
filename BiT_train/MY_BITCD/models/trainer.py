import numpy as np                # 导入数值计算库
import matplotlib.pyplot as plt    # 导入绘图库
import os                          # 导入操作系统功能库
import datetime                  # 导入日期时间库
import traceback

import utils                       # 导入工具函数库
from models.networks import *      # 导入网络模型定义

import torch                       # 导入PyTorch
import torch.optim as optim        # 导入优化器
import torch.nn.functional as F     # 导入F函数

from misc.metric_tool import ConfuseMatrixMeter  # 导入混淆矩阵计算工具
from models.losses import cross_entropy          # 导入交叉熵损失函数
import models.losses as losses                   # 导入所有损失函数

from misc.logger_tool import Logger, Timer       # 导入日志和计时器工具

from utils import de_norm                        # 导入反归一化函数


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
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)  # 从networks.py中调用define_G函数初始化网络

        # 设置计算设备（GPU或CPU）
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # 学习率设置
        self.lr = args.lr

        # 定义优化器，使用SGD优化器
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,            # 动量参数
                                     weight_decay=5e-4)       # L2正则化系数

        # 定义学习率调度器
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        # 初始化评估指标计算器（混淆矩阵）
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # 设置日志文件
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)  # 记录所有配置参数
        
        # 初始化计时器
        self.timer = Timer()
        self.batch_size = args.batch_size

        # 初始化训练日志相关变量
        self.epoch_acc = 0                 # 当前epoch的准确率
        self.best_val_acc = 0.0            # 最佳验证集准确率
        self.best_epoch_id = 0             # 最佳epoch的ID
        self.epoch_to_start = 0            # 起始epoch（用于断点续训）
        self.max_num_epochs = args.max_epochs  # 最大训练epoch数

        # 训练步数计算
        self.global_step = 0               # 全局训练步数
        self.steps_per_epoch = len(dataloaders['train'])  # 每个epoch的步数
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch  # 总步数

        # 存储中间结果的变量
        self.G_pred = None                 # 模型预测结果
        self.pred_vis = None               # 预测结果可视化
        self.batch = None                  # 当前批次数据
        self.G_loss = None                 # 生成器损失
        self.is_training = False           # 是否处于训练模式
        self.batch_id = 0                  # 当前批次ID
        self.epoch_id = 0                  # 当前epoch ID
        self.checkpoint_dir = args.checkpoint_dir  # 检查点保存路径
        self.vis_dir = args.vis_dir        # 可视化结果保存路径
        self.edge1 = None                  # 第一帧边缘检测结果
        self.edge2 = None                  # 第二帧边缘检测结果
        self.edge_loss = 0.0               # 边缘损失

        # 根据参数选择损失函数
        if args.loss == 'ce':              # 交叉熵损失
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':           # 二元交叉熵损失
            pass  # 使用自定义的_pxl_loss方法
        else:
            raise NotImplemented(args.loss)
            
        # 配置损失函数权重（改为动态初始化）
        self.class_weights = torch.tensor([1.0, 50.0]).to(self.device)  # 初始权重
        self.edge_lambda = 0.1  # 边缘损失初始权重
        
        # 添加自适应损失参数
        self.use_adaptive_weights = True  # 是否使用自适应权重
        self.loss_weight_update_freq = 10  # 每隔多少批次更新一次权重
        self.class_pixels_count = torch.zeros(2).to(self.device)  # 类别像素计数
        self.total_samples = 0  # 样本总数，用于计算移动平均
        self.weight_smooth_factor = 0.9  # 平滑因子，用于权重平滑更新
        self.edge_weight_min = 0.05  # 边缘损失最小权重
        self.edge_weight_max = 0.2  # 边缘损失最大权重
        self.edge_weight_warmup = True  # 边缘损失预热
        
        # 记录各类损失值，用于分析
        self.ce_loss_value = 0.0
        self.focal_loss_value = 0.0
        self.edge_loss_value = 0.0
        
        # 初始化/加载验证集准确率历史记录
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        
        # 初始化/加载训练集准确率历史记录
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # 检查并创建模型保存目录
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # 设置图像大小，用于可视化和错误处理
        self.img_size = args.img_size if hasattr(args, 'img_size') else 256

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        """
        加载检查点，用于继续训练或推理
        
        参数:
            ckpt_name: 检查点文件名
        """
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # 加载整个检查点
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # 更新网络G的状态
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            # 更新优化器状态
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            # 更新学习率调度器状态
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            # 将模型加载到设备上
            self.net_G.to(self.device)

            # 更新其他训练状态
            self.epoch_to_start = checkpoint['epoch_id'] + 1  # 设置起始epoch为上次保存的下一个
            self.best_val_acc = checkpoint['best_val_acc']    # 加载最佳验证准确率
            self.best_epoch_id = checkpoint['best_epoch_id']  # 加载最佳epoch ID

            # 重新计算总步数
            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')  # 如果检查点不存在，从头开始训练

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

    def _visualize_pred(self):
        """
        将预测结果转换为可视化图像
        
        返回:
            pred_vis: 预测结果的可视化（255表示变化，0表示不变）
        """
        # 对预测结果进行处理
        if self.G_pred.shape[1] == 1:  # 单通道输出
            pred = (torch.sigmoid(self.G_pred) > 0.5).long()
        else:  # 多通道输出
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
            
        # 将1（变化）映射为255，便于可视化
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        """
        保存检查点
        
        参数:
            ckpt_name: 检查点文件名
        """
        # 保存模型状态、优化器状态、训练状态等
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))
        
        # 单独保存模型权重(state_dict)，以提高跨环境兼容性
        weights_path = os.path.join(self.checkpoint_dir, ckpt_name.replace('.pt', '_weights.pth'))
        torch.save(self.net_G.state_dict(), weights_path)
        self.logger.write(f'保存模型权重到: {weights_path}\n')

    def _update_lr_schedulers(self):
        """
        更新学习率调度器，通常在每个epoch结束时调用
        """
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        计算并更新度量指标
        """
        # 获取模型预测结果
        G_pred = self.G_pred.detach()
        target = self.batch['L'].to(self.device).detach()

        # 确保预测结果和标签的空间尺寸匹配
        if G_pred.shape[-2:] != target.shape[-2:]:
            G_pred = F.interpolate(G_pred, target.shape[-2:], mode='nearest')
        
        # 确保target具有正确的形状
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B,1,H,W] -> [B,H,W]
        
        # 获取预测类别
        if G_pred.shape[1] == 1:  # 单通道输出
            # 应用sigmoid并二值化
            G_pred_cls = (torch.sigmoid(G_pred.squeeze(1)) > 0.5).long()
        else:  # 多通道输出
            # 取argmax作为预测类别
            G_pred_cls = torch.argmax(G_pred, dim=1)
        
        # 更新混淆矩阵
        np_pred = G_pred_cls.cpu().numpy()
        np_target = target.cpu().numpy()
        current_score = self.running_metric.update_cm(pr=np_pred, gt=np_target)
        return current_score

    def _collect_running_batch_states(self):
        """
        收集并记录当前批次的训练状态
        """
        # 更新指标并获取当前得分
        running_acc = self._update_metric()

        # 确定数据集大小
        m = len(self.dataloaders['train'])
        if not self.is_training:
            m = len(self.dataloaders['val'])

        # 更新计时器
        imps, est = self._timer_update()
        
        # 定期打印训练状态（每10个批次）
        if np.mod(self.batch_id, 10) == 1:
            # 添加各部分损失信息
            loss_info = f"| CE: {self.ce_loss_value:.4f} | Focal: {self.focal_loss_value:.4f}"
            if self.edge_loss_value > 0:
                loss_info += f" | 边缘({self.edge_lambda:.3f}): {self.edge_loss_value:.4f}"
            
            message = f'状态: {"训练" if self.is_training else "验证"} | Epoch [{self.epoch_id}/{self.max_num_epochs-1}] | 批次 [{self.batch_id}/{m}] | 速度: {imps*self.batch_size:.2f} img/s | 剩余: {est:.2f}h | 总损失: {self.G_loss.item():.5f} {loss_info} | F1: {running_acc:.5f}'
            self.logger.write(message + '\n')
            # 打印到控制台，确保可见
            print(message)
            
            # 每10个批次生成一次可视化图像
            self._save_visualization()

    def _collect_epoch_states(self):
        """
        收集并记录当前epoch的训练状态
        """
        # 获取当前epoch的评估指标
        scores = self.running_metric.get_scores()
        # 使用mean F1 score作为主要指标
        self.epoch_acc = scores['mf1']
        
        # 记录当前epoch的状态
        message = f'当前阶段: {"训练" if self.is_training else "验证"} | Epoch {self.epoch_id}/{self.max_num_epochs-1} | '
        message += f'mF1: {scores["mf1"]:.5f} | mIoU: {scores["miou"]:.5f} | Accuracy: {scores["acc"]:.5f} | '
        message += f'变化F1: {scores["F1_1"]:.5f} | 变化IoU: {scores["iou_1"]:.5f}'
        
        self.logger.write(message + '\n\n')
        print(message)

    def _update_checkpoints(self):
        """
        更新检查点，保存当前模型和最佳模型
        """
        # 保存当前模型
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # 如果当前模型优于历史最佳，更新最佳模型
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

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
        self.running_metric.clear()

    def _forward_pass(self, batch):
        """
        执行前向传播
        
        参数:
            batch: 当前批次数据
        """
        self.batch = batch
        
        # 获取输入图像并转移到设备上
        img_in1 = batch['A'].to(self.device)  # 时间1的图像
        img_in2 = batch['B'].to(self.device)  # 时间2的图像
        
        # 执行模型前向传播
        outputs = self.net_G(img_in1, img_in2)
        
        # 处理网络输出
        if isinstance(outputs, tuple):
            if len(outputs) == 2:  # (预测, 边缘图)
                self.G_pred, self.edge_maps = outputs
                self.edge1 = self.edge2 = self.edge_maps
            elif len(outputs) >= 3:  # (预测, 边缘1, 边缘2)
                self.G_pred, self.edge1, self.edge2 = outputs[:3]
        else:
            self.G_pred = outputs
            self.edge1 = self.edge2 = None

    def _backward_G(self):
        """
        执行反向传播计算梯度
        """
        # 获取当前批次的真实标签
        target = self.batch['L'].to(self.device)
        
        # 确保target具有正确的数据类型
        if target.dtype != torch.long:
            target = target.long()
        
        # 确保标签维度正确
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # 从[B,1,H,W]变为[B,H,W]
        
        # 更新类别像素统计
        if self.use_adaptive_weights:
            self._update_class_statistics(target)
        
        # 主任务损失 - 变化检测
        self.G_loss, self.ce_loss_value, self.focal_loss_value = self._pxl_loss(self.G_pred, target)
        
        # 边缘检测辅助损失 - 只有当边缘检测输出不为None时才计算
        if self.edge1 is not None and self.edge2 is not None:
            # 生成边缘标签
            img1 = self.batch['A'].to(self.device)
            img2 = self.batch['B'].to(self.device)
            
            # 灰度化图像
            img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
            
            # 使用Sobel算子计算边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
            
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            # 批量计算边缘
            pad_mode = 'reflect'
            edge_x1 = F.conv2d(F.pad(img1_gray.unsqueeze(1), [1, 1, 1, 1], mode=pad_mode), sobel_x)
            edge_y1 = F.conv2d(F.pad(img1_gray.unsqueeze(1), [1, 1, 1, 1], mode=pad_mode), sobel_y)
            edge_x2 = F.conv2d(F.pad(img2_gray.unsqueeze(1), [1, 1, 1, 1], mode=pad_mode), sobel_x)
            edge_y2 = F.conv2d(F.pad(img2_gray.unsqueeze(1), [1, 1, 1, 1], mode=pad_mode), sobel_y)
            
            # 计算梯度幅值并归一化
            edge_mag1 = torch.sqrt(edge_x1**2 + edge_y1**2)
            edge_mag2 = torch.sqrt(edge_x2**2 + edge_y2**2)
            
            # 批量归一化 - 每个样本单独归一化
            for i in range(edge_mag1.size(0)):
                if edge_mag1[i].max() > 0:
                    edge_mag1[i] = edge_mag1[i] / edge_mag1[i].max()
                if edge_mag2[i].max() > 0:
                    edge_mag2[i] = edge_mag2[i] / edge_mag2[i].max()
            
            # 调整边缘标签大小以匹配预测
            if edge_mag1.shape[-2:] != self.edge1.shape[-2:]:
                edge_mag1 = F.interpolate(edge_mag1, size=self.edge1.shape[-2:], mode='bilinear', align_corners=False)
                edge_mag2 = F.interpolate(edge_mag2, size=self.edge2.shape[-2:], mode='bilinear', align_corners=False)
            
            # 计算边缘损失
            edge_loss1 = F.mse_loss(self.edge1, edge_mag1)
            edge_loss2 = F.mse_loss(self.edge2, edge_mag2)
            edge_loss = 0.5 * (edge_loss1 + edge_loss2)
            
            # 自适应边缘损失权重
            if self.use_adaptive_weights:
                self._update_edge_lambda(edge_loss.item())
            
            # 添加边缘损失到总损失
            self.G_loss = self.G_loss + self.edge_lambda * edge_loss
            self.edge_loss_value = edge_loss.item()
        else:
            self.edge_loss_value = 0.0
        
        # 反向传播
        self.G_loss.backward()

    def _update_class_statistics(self, target):
        """
        更新类别像素统计，用于自适应损失权重
        
        参数:
            target: 真实标签 [B, H, W]
        """
        # 计算当前批次中每个类别的像素数量
        batch_class_count = torch.zeros(2).to(self.device)
        for c in range(2):
            batch_class_count[c] = (target == c).sum().float()
        
        # 更新累积统计
        self.class_pixels_count = self.weight_smooth_factor * self.class_pixels_count + \
                                 (1 - self.weight_smooth_factor) * batch_class_count
        self.total_samples += target.numel()
        
        # 每隔一定批次更新权重
        if self.batch_id % self.loss_weight_update_freq == 0 and self.total_samples > 0:
            # 防止除零
            eps = 1e-5
            # 计算类别频率
            class_freq = self.class_pixels_count / (self.class_pixels_count.sum() + eps)
            # 频率越低，权重越高（逆频率权重）
            inverse_freq = 1.0 / (class_freq + eps)
            # 归一化权重，确保均值为1
            normalized_weights = inverse_freq / (inverse_freq.mean() + eps)
            # 限制权重范围，避免过大的不平衡
            min_weight, max_weight = 1.0, 50.0
            normalized_weights = torch.clamp(normalized_weights, min=min_weight, max=max_weight)
            # 平滑更新权重
            self.class_weights = self.weight_smooth_factor * self.class_weights + \
                               (1 - self.weight_smooth_factor) * normalized_weights
            
            # 每个epoch结束时打印权重信息
            if self.batch_id == 0:
                self.logger.write(f"当前类别权重: 无变化={self.class_weights[0]:.2f}, 变化={self.class_weights[1]:.2f}\n")
    
    def _update_edge_lambda(self, edge_loss):
        """
        更新边缘损失权重
        
        参数:
            edge_loss: 当前的边缘损失值
        """
        # 边缘损失预热策略
        if self.edge_weight_warmup:
            # 在前10%的epoch中逐渐增加边缘损失权重
            progress = min(1.0, self.epoch_id / (0.1 * self.max_num_epochs))
            base_lambda = self.edge_weight_min + progress * (self.edge_weight_max - self.edge_weight_min)
        else:
            base_lambda = self.edge_weight_max
        
        # 根据当前损失值动态调整
        # 损失值越大，适当降低权重避免主任务受影响
        loss_factor = 1.0 / (1.0 + edge_loss)
        
        # 最终边缘损失权重
        target_lambda = base_lambda * loss_factor
        target_lambda = max(self.edge_weight_min, min(self.edge_weight_max, target_lambda))
        
        # 平滑更新
        self.edge_lambda = self.weight_smooth_factor * self.edge_lambda + \
                         (1 - self.weight_smooth_factor) * target_lambda

    def _pxl_loss(self, pred, target):
        """
        计算像素级损失函数，使用带权重的交叉熵损失和Focal Loss
        
        参数:
            pred: 模型预测，形状可能是[B,1,H,W]或[B,2,H,W]
            target: 真实标签，形状为[B,H,W]
            
        返回:
            total_loss: 总损失
            ce_loss_val: 交叉熵损失值
            focal_loss_val: Focal Loss损失值
        """
        # 确保target类型正确
        if target.dtype != torch.long:
            target = target.long()
            
        # 确保target的正确形状
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # 从[B,1,H,W]变为[B,H,W]
            
        # 确保预测结果和标签的空间尺寸匹配
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, target.shape[-2:], mode='bilinear', align_corners=False)
        
        # 处理不同类型的预测输出
        if pred.shape[1] == 1:  # 单通道输出[B,1,H,W]
            # 应用sigmoid并计算BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(
                pred.squeeze(1), target.float(), reduction='none'
            )
            
            # 计算Focal Loss权重
            pt = torch.exp(-bce_loss)
            focal_weight = (1 - pt) ** 2.0  # gamma=2.0
            
            # 应用类别权重
            pos_weight = self.class_weights[1] / self.class_weights[0]
            alpha = target.float() * pos_weight + (1 - target.float())
            
            # 完整的Focal Loss
            focal_loss = alpha * focal_weight * bce_loss
            
            ce_loss_val = bce_loss.mean().item()
            focal_loss_val = focal_loss.mean().item()
            
            return focal_loss.mean(), ce_loss_val, focal_loss_val
                
        else:  # 多通道输出[B,C,H,W]
            # 使用交叉熵损失
            ce_loss = F.cross_entropy(
                pred, target, 
                weight=self.class_weights,
                reduction='none'
            )
            
            # 计算Focal Loss权重
            pred_softmax = F.softmax(pred, dim=1)
            batch_size = pred.shape[0]
            
            # 获取每个样本对应类别的预测概率 - 优化版本，减少循环
            # 创建类别索引掩码
            classes = torch.arange(pred.shape[1], device=pred.device)
            
            # 扩展target和类别以匹配形状
            target_expand = target.view(batch_size, 1, -1)  # [B, 1, H*W]
            classes_expand = classes.view(1, -1, 1)  # [1, C, 1]
            
            # 创建匹配掩码
            match = (target_expand == classes_expand)  # [B, C, H*W]
            
            # 展平预测概率
            pred_flat = pred_softmax.view(batch_size, -1, pred.shape[2] * pred.shape[3])  # [B, C, H*W]
            
            # 使用掩码获取对应类别概率
            probs_masked = torch.where(match, pred_flat, torch.zeros_like(pred_flat))
            
            # 按类别求和并重塑
            probs = torch.sum(probs_masked, dim=1).view(batch_size, *target.shape[1:])  # [B, H, W]
            
            # 计算Focal Loss权重
            focal_weight = (1 - probs) ** 2.0  # gamma=2.0
            
            # 计算最终损失
            focal_loss = focal_weight * ce_loss
            
            ce_loss_val = ce_loss.mean().item()
            focal_loss_val = focal_loss.mean().item()
            
            return focal_loss.mean(), ce_loss_val, focal_loss_val

    def _save_visualization(self):
        """
        生成并保存训练过程的可视化图像，按以下排列：
        第一排：第一时间点的原始图像
        第二排：第二时间点的原始图像
        第三排：真实标签
        第四排：模型预测
        """
        try:
            # 反归一化输入图像
            vis_input1 = utils.make_numpy_grid(de_norm(self.batch['A']))  # 第一时间点图像
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))  # 第二时间点图像

            # 获取预测结果可视化
            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            # 处理真实标签进行可视化
            label_tensor = self.batch['L'].clone()
            
            # 特殊处理[b,1,h,w,3]格式的标签
            if label_tensor.dim() == 5 and label_tensor.shape[-1] == 3:
                label_tensor = label_tensor[:, :, :, :, 0]
            
            # 确保标签维度正确，便于可视化
            if label_tensor.dim() > 4:
                label_tensor = label_tensor.squeeze(1)  # 去掉多余维度
            if label_tensor.dim() == 4 and label_tensor.size(1) > 1:
                # 如果有多个通道，只保留一个
                label_tensor = label_tensor[:, 0:1, :, :]
            if label_tensor.dim() == 3:
                # 如果是[B,H,W]格式，添加通道维度
                label_tensor = label_tensor.unsqueeze(1)

            # 标签可视化
            vis_gt = utils.make_numpy_grid(label_tensor * 255)  # 乘以255使得标签更清晰可见

            # 按顺序垂直堆叠所有可视化结果（保持顺序：输入1、输入2、真实标签、预测）
            vis = np.concatenate([vis_input1, vis_input2, vis_gt, vis_pred], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)  # 裁剪到[0,1]范围
            
            # 根据训练/验证阶段命名
            phase = 'train' if self.is_training else 'val'
            
            # 创建文件名，包含epoch和batch信息
            file_name = os.path.join(
                self.vis_dir, f'{phase}_e{self.epoch_id}_b{self.batch_id}.jpg')
            
            # 保存可视化图像
            plt.imsave(file_name, vis)
            
            return vis
        except Exception as e:
            print(f"可视化生成错误: {e}")
            traceback.print_exc()
            return None

    def train_models(self):
        """
        训练模型的主函数
        """
        # 尝试加载检查点
        self._load_checkpoint()

        # 在数据集上循环多个epoch
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()  # 清除上一个epoch的缓存
            self.is_training = True
            self.net_G.train()  # 设置模型为训练模式
            
            # 记录当前学习率
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            
            # 遍历训练数据
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)  # 前向传播
                
                # 更新生成器
                self.optimizer_G.zero_grad()  # 清除梯度
                self._backward_G()            # 反向传播
                self.optimizer_G.step()       # 优化器更新参数
                
                self._collect_running_batch_states()  # 收集批次状态
                self._timer_update()                  # 更新计时器

            # 收集整个epoch的状态
            self._collect_epoch_states()
            # 更新训练准确率曲线
            self._update_training_acc_curve()
            # 更新学习率
            self._update_lr_schedulers()


            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()  # 清除训练阶段的缓存
            self.is_training = False
            self.net_G.eval()    # 设置模型为评估模式

            # 遍历验证数据
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():  # 不计算梯度
                    self._forward_pass(batch)  # 前向传播
                self._collect_running_batch_states()  # 收集批次状态
                
            # 收集整个验证阶段的状态
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            # 更新验证准确率曲线
            self._update_val_acc_curve()
            # 更新检查点
            self._update_checkpoints()

