import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.networks import *         # 导入网络模型定义
from misc.metric_tool import ConfuseMatrixMeter  # 导入混淆矩阵计算工具
from misc.logger_tool import Logger    # 导入日志工具
from utils import de_norm             # 导入反归一化函数
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():
    """
    变化检测模型评估器
    用于加载训练好的模型并在测试集上进行评估
    """
    def __init__(self, args, dataloader):
        """
        初始化评估器
        
        参数:
            args: 配置参数
            dataloader: 测试数据加载器
        """
        self.dataloader = dataloader

        self.n_class = args.n_class  # 类别数量
        # 定义生成器网络G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)  # 从networks.py中调用define_G函数初始化网络
        # 设置计算设备（GPU或CPU）
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # 初始化评估指标计算器
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # 设置测试日志文件
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)  # 记录所有配置参数


        # 初始化评估相关变量
        self.epoch_acc = 0             # 当前评估的准确率
        self.best_val_acc = 0.0        # 最佳验证集准确率(从检查点加载)
        self.best_epoch_id = 0         # 最佳epoch的ID(从检查点加载)

        self.steps_per_epoch = len(dataloader)  # 每个epoch的步数

        # 存储中间结果的变量
        self.G_pred = None             # 模型预测结果
        self.pred_vis = None           # 预测结果可视化
        self.batch = None              # 当前批次数据
        self.is_training = False       # 始终为False，表示评估模式
        self.batch_id = 0              # 当前批次ID
        self.epoch_id = 0              # 当前epoch ID
        self.checkpoint_dir = args.checkpoint_dir  # 检查点目录
        self.vis_dir = args.vis_dir    # 可视化结果保存目录

        # 检查并创建必要的目录
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        """
        加载模型检查点
        
        参数:
            checkpoint_name: 检查点文件名，默认为'best_ckpt.pt'
        """
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # 加载整个检查点
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            # 加载模型参数
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            # 将模型加载到设备上
            self.net_G.to(self.device)

            # 更新最佳模型记录
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            # 尝试加载完整模型
            full_model_path = os.path.join(self.checkpoint_dir, checkpoint_name.replace('.pt', '_full_model.pth'))
            if os.path.exists(full_model_path):
                self.logger.write('loading full model checkpoint...\n')
                # 直接加载完整模型
                self.net_G = torch.load(full_model_path, map_location=self.device)
                self.logger.write('Successfully loaded full model from %s\n' % full_model_path)
                self.logger.write('\n')
            else:
                raise FileNotFoundError('no such checkpoint %s or %s' % (checkpoint_name, full_model_path))

    def load_full_model(self, model_path):
        """
        直接加载完整模型，用于推理
        
        参数:
            model_path: 完整模型文件路径
        """
        if os.path.exists(model_path):
            self.logger.write('loading full model checkpoint...\n')
            # 直接加载完整模型
            self.net_G = torch.load(model_path, map_location=self.device)
            self.logger.write('Successfully loaded full model from %s\n' % model_path)
            self.logger.write('\n')
            return True
        else:
            self.logger.write('Failed to load model: %s does not exist\n' % model_path)
            return False

    def _visualize_pred(self):
        """
        将预测结果转换为可视化图像
        
        返回:
            pred_vis: 预测结果的可视化（255表示变化，0表示不变）
        """
        # 对预测结果进行argmax得到类别索引
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # 将1（变化）映射为255，便于可视化
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        更新评估指标
        
        返回:
            current_score: 当前批次的评估得分（F1分数）
        """
        # 获取当前批次的真实标签
        target = self.batch['L'].to(self.device).detach()
        # 获取模型预测结果
        G_pred = self.G_pred.detach()
        # 将预测概率转换为类别索引
        G_pred = torch.argmax(G_pred, dim=1)
        
        # 只在第一个批次打印基本信息
        if self.batch_id == 0:
            print(f"测试开始 - 批处理大小: {target.size(0)}, 图像尺寸: {target.shape[-2:]}")
            
        # 确保预测结果和标签的空间尺寸匹配
        if G_pred.shape[-2:] != target.shape[-2:]:
            G_pred = F.interpolate(
                G_pred.unsqueeze(1).float(), 
                size=target.shape[-2:], 
                mode='nearest'
            ).squeeze(1).long()

        # 更新混淆矩阵并获取当前得分
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):
        """
        收集并记录当前批次的评估状态，包括指标更新和可视化
        """
        # 更新指标并获取当前得分
        running_acc = self._update_metric()

        # 总批次数
        m = len(self.dataloader)

        # 定期打印评估状态
        if np.mod(self.batch_id, 100) == 1:
            message = '评估进度: [%d/%d] | 当前mF1: %.5f\n' % (self.batch_id, m, running_acc)
            self.logger.write(message)
            print(message, end='')

        # 定期生成可视化结果
        if np.mod(self.batch_id, 100) == 1:
            try:
                # 反归一化输入图像
                vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))     # 时间1的图像
                vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))    # 时间2的图像

                # 模型预测结果可视化
                vis_pred = utils.make_numpy_grid(self._visualize_pred())

                # 标签可视化前先检查维度
                label_tensor = self.batch['L'].clone()
                
                # 特殊处理[b,1,h,w,3]格式的标签
                if label_tensor.dim() == 5 and label_tensor.shape[-1] == 3:
                    # 取第一个通道，去掉RGB维度
                    label_tensor = label_tensor[:, :, :, :, 0]
                
                # 确保标签可被可视化
                if label_tensor.dim() > 4:
                    label_tensor = label_tensor.squeeze(1)  # 去掉多余维度
                if label_tensor.dim() == 4 and label_tensor.size(1) > 1:
                    # 如果有多个通道，只保留一个
                    label_tensor = label_tensor[:, 0:1, :, :]
                if label_tensor.dim() == 3:
                    # 如果是[B,H,W]格式，添加通道维度
                    label_tensor = label_tensor.unsqueeze(1)

                # 真实标签可视化
                vis_gt = utils.make_numpy_grid(label_tensor)
                # 连接所有可视化结果
                vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)  # 裁剪到[0,1]范围
                # 保存可视化图像
                file_name = os.path.join(
                    self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
                plt.imsave(file_name, vis)
            except Exception as e:
                print(f"评估可视化错误，继续评估")
                # 继续评估，不因可视化错误而中断


    def _collect_epoch_states(self):
        """
        收集并记录整个评估过程的状态和指标
        """
        # 获取所有评估指标
        scores_dict = self.running_metric.get_scores()

        # 保存评估指标到文件
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        # 使用平均F1分数作为主要评估指标
        self.epoch_acc = scores_dict['mf1']

        # 创建以准确率命名的文件，用于标记最终性能
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        # 打印评估结果
        print("\n========== 测试评估结果 ==========")
        # 创建详细的指标表格形式输出
        summary = (
            f"| 主要指标 | mF1={scores_dict['mf1']:.5f} | mIoU={scores_dict['miou']:.5f} | Accuracy={scores_dict['acc']:.5f} |\n"
            f"| 变化类别 | 精确率={scores_dict['precision_1']:.5f} | 召回率={scores_dict['recall_1']:.5f} | F1={scores_dict['F1_1']:.5f} | IoU={scores_dict['iou_1']:.5f} |\n"
            f"| 背景类别 | 精确率={scores_dict['precision_0']:.5f} | 召回率={scores_dict['recall_0']:.5f} | F1={scores_dict['F1_0']:.5f} | IoU={scores_dict['iou_0']:.5f} |"
        )
        print(summary)
        print("===============================\n")
            
        # 记录所有评估指标
        self.logger.write("========== 详细评估结果 ==========\n")
        self.logger.write(summary + "\n\n")
        
        # 保存所有细节指标到日志
        metrics_detail = '所有评估指标:\n'
        for k, v in scores_dict.items():
            metrics_detail += '%s: %.5f\n' % (k, v)
        self.logger.write(metrics_detail+'\n\n')

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
        self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        """
        评估模型的主函数
        
        参数:
            checkpoint_name: 要加载的检查点文件名，默认为'best_ckpt.pt'
        """
        # 加载检查点
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()  # 清除缓存
        self.is_training = False
        self.net_G.eval()  # 设置模型为评估模式

        # 遍历测试数据
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():  # 不计算梯度
                self._forward_pass(batch)  # 前向传播
            self._collect_running_batch_states()  # 收集批次状态
        self._collect_epoch_states()  # 收集整个评估阶段的状态
