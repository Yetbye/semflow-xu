import torch
import numpy as np
from pathlib import Path

class ClassificationEvaluator:
    def __init__(self, output_dir, threshold=0.5):
        """
        初始化分类评估器
        :param output_dir: 结果保存目录
        :param threshold: 多标签分类的阈值，默认 0.5
        """
        self.threshold = threshold
        self.total_correct = 0
        self.total_samples = 0
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开日志文件
        self.log_file_path = self.output_dir / "cls_predictions.txt"
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        header = f"{'Batch':<6} | {'Idx':<4} | {'Ground Truth Indices':<30} | {'Pred Indices':<30} | {'Pred Probs'}\n"
        self.log_file.write(header)
        self.log_file.write("-" * 120 + "\n")

    def update(self, cls_logits, cls_target, batch_idx):
        """
        更新评估指标并写入日志
        :param cls_logits: 模型输出的 logits (B, num_classes)
        :param cls_target: 真实标签 (B, num_classes)
        :param batch_idx: 当前 batch 的索引
        """
        # 确保数据在 CPU 上
        logits = cls_logits.detach().cpu()
        targets = cls_target.detach().cpu()
        
        # Sigmoid -> 概率
        probs = torch.sigmoid(logits)
        # 阈值 -> 0/1 预测
        preds = (probs > self.threshold).float()
        
        # 计算准确率 (Element-wise Accuracy: 每个类别的预测都算一次)
        # 如果需要 Exact Match (全对才算对)，逻辑需改为 (preds == targets).all(dim=1).sum()
        correct = (preds == targets).all(dim=1).float().sum()
        total = targets.size(0) 
        
        self.total_correct += correct
        self.total_samples += total
        
        # 写入日志
        self._log_batch(probs.numpy(), preds.numpy(), targets.numpy(), batch_idx)

    def _log_batch(self, probs_np, preds_np, targets_np, batch_idx):
        bsz = probs_np.shape[0]
        for i in range(bsz):
            # 获取非零元素的索引 (即存在的类别 ID)
            gt_indices = np.where(targets_np[i] == 1)[0]
            pred_indices = np.where(preds_np[i] == 1)[0]
            
            # 获取预测类别的置信度
            pred_probs_str = [f"{probs_np[i][idx]:.2f}" for idx in pred_indices]
            
            # 格式化字符串
            gt_str = str(gt_indices.tolist())
            pred_str = str(pred_indices.tolist())
            probs_str = str(pred_probs_str)
            
            log_line = f"{batch_idx:<6} | {i:<4} | {gt_str:<30} | {pred_str:<30} | {probs_str}\n"
            self.log_file.write(log_line)
        
        self.log_file.flush() # 实时写入磁盘

    def compute(self):
        """
        计算最终准确率
        """
        if self.total_samples == 0:
            acc = 0.0
        else:
            acc = self.total_correct / self.total_samples
        
        # 【修改】将最终结果写入日志文件
        self.log_file.write("\n" + "=" * 120 + "\n")
        self.log_file.write(f"Final Classification Accuracy (Exact Match): {acc:.4f}\n")
        self.log_file.write(f"Total Samples: {self.total_samples}, Correct Samples: {int(self.total_correct)}\n")
        self.log_file.write("=" * 120 + "\n")
        self.log_file.flush()

        return acc

    def close(self):
        """
        关闭文件句柄
        """
        if self.log_file:
            self.log_file.close()