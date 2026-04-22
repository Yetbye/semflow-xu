import torch

class ClassificationEvaluator:
    def __init__(self, threshold=0.5):
        """
        初始化纯内存分类评估器 (不生成任何文件)
        :param threshold: 多标签分类的判定阈值，默认 0.5
        """
        self.threshold = threshold
        self.total_correct = 0
        self.total_samples = 0

    def update(self, cls_logits, cls_target):
        """
        累加当前批次的预测结果
        :param cls_logits: 模型输出 (B, num_classes)
        :param cls_target: 真实标签 (B, num_classes)
        """
        logits = cls_logits.detach().cpu()
        targets = cls_target.detach().cpu()
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        targets = targets.int()

        # 计算所有元素级别的准确率
        self.total_correct += (preds == targets).sum().item()
        self.total_samples += targets.numel()

    def compute(self):
        """返回当前的分类准确率"""
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    def close(self):
        """占位符，保持与原接口的兼容性"""
        pass