import numpy as np

class BinaryIoU:
    """
    Computes Binary Intersection-over-Union matrix.
    Treats class 0 as Background, and all classes > 0 as a single Foreground class.
    Interface is compatible with the original IoU class.
    """

    def __init__(self, num_classes, ignore_index=None):
        # 虽然传入了原始类别数（如12），但我们内部只关心 2 类：0(背景) 和 1(前景)
        self.original_num_classes = num_classes 
        self.num_classes = 2 
        self.ignore_index = ignore_index
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def add(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)

        # 1. 基于原始标签创建有效掩码
        # 过滤掉 ignore_index 以及超出原始类别范围的异常值
        valid_mask = (target != self.ignore_index) & \
                     (target < self.original_num_classes) & \
                     (pred < self.original_num_classes)
        
        # 2. 提取有效像素
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

        # 3. 【核心修改】二值化处理
        # 0 保持为 0 (背景)
        # >0 的所有类别 (1-11) 全部变为 1 (前景)
        valid_pred_bin = (valid_pred > 0).astype(int)
        valid_target_bin = (valid_target > 0).astype(int)

        # 4. 更新混淆矩阵 (大小固定为 2x2)
        self.hist += np.bincount(
            self.num_classes * valid_target_bin + valid_pred_bin,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def get_iou(self):
        # 计算 2 个类别的 IoU: [IoU_background, IoU_foreground]
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iou

    def get_miou(self):
        iou = self.get_iou()
        miou = np.nanmean(iou)
        return miou