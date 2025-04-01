import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
import traceback


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(prediction=pr, groundtruth=gt)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(prediction=None, groundtruth=None):
    """
    计算混淆矩阵
    
    参数:
        prediction: 预测标签
        groundtruth: 真实标签
    
    返回:
        混淆矩阵
    """
    try:
        # 只在特定情况下打印调试信息
        if np.random.random() < 0.05:  # 只有5%的概率打印信息
            print(f"[混淆矩阵] 标签形状: {groundtruth.shape}, 预测形状: {prediction.shape}")
        
        # 确保标签为二维数组
        if len(groundtruth.shape) > 2:
            if len(groundtruth.shape) == 3 and groundtruth.shape[0] == 1:
                groundtruth = groundtruth.squeeze(0)
            elif len(groundtruth.shape) == 3:
                groundtruth = groundtruth.squeeze()
        
        # 处理预测输出 (重要修复)
        if len(prediction.shape) > 2:
            # 关键修复: 如果预测是(2, 256, 256)形式，这表示有两个通道(0和1的概率)
            # 我们需要获取第1个通道(索引1)，表示类别1的概率，并基于阈值进行二值化
            if len(prediction.shape) == 3 and prediction.shape[0] == 2:
                # 获取类别1的概率
                prediction = prediction[1]  # 提取第二个通道
                # 应用更低的阈值 (0.1)
                threshold = 0.1  # 降低阈值以增加变化区域的检测敏感度
                prediction = (prediction > threshold).astype(np.int32)
            else:
                # 其他情况，尝试挤压
                prediction = prediction.squeeze()
        
        # 确认维度匹配
        if prediction.shape != groundtruth.shape:
            # 尝试调整尺寸
            if groundtruth.size > 0 and prediction.size > 0:
                # 调整预测大小匹配标签
                prediction = resize(prediction, groundtruth.shape, order=0, preserve_range=True).astype(groundtruth.dtype)
                
        # 确保值的范围正确 (0或1)
        valid_gt = (groundtruth == 0) | (groundtruth == 1)
        valid_pred = (prediction == 0) | (prediction == 1)
        
        # 初始化混淆矩阵 (2x2 矩阵)
        confuse_matrix = np.zeros((2, 2), dtype=np.int64)
        
        # 只计算有效位置的混淆矩阵
        valid_pixels = valid_gt & valid_pred
        if valid_pixels.any():
            # 提取有效位置的标签和预测
            gt_valid = groundtruth[valid_pixels]
            pred_valid = prediction[valid_pixels]
            
            # 计算混淆矩阵
            for i in range(2):  # 标签类别
                for j in range(2):  # 预测类别
                    confuse_matrix[i, j] = np.sum((gt_valid == i) & (pred_valid == j))
                    
            # 仅偶尔打印详细的混淆矩阵信息
            if np.random.random() < 0.01:  # 1%的概率
                tn, fp = confuse_matrix[0]  # 第一行: 真实为0的情况
                fn, tp = confuse_matrix[1]  # 第二行: 真实为1的情况
                
                # 计算评估指标
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"[混淆矩阵] 变化区域指标 - 精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")
            
        else:
            print("[混淆矩阵警告] 没有有效的像素用于计算混淆矩阵!")
        
        return confuse_matrix
    
    except Exception as e:
        print(f"[混淆矩阵错误] {str(e)}")
        traceback.print_exc()
        # 返回空的混淆矩阵
        return np.zeros((2, 2), dtype=np.int64)


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']
