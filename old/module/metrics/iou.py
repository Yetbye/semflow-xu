import numpy as np

class IoU:
    """
    Computes intersection-over-union matrix.
    """

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes))

    def add(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)

        # Create a mask for valid pixels.
        # A pixel is valid if its target is not the ignore_index,
        # and both target and prediction are within the number of classes.
        valid_mask = (target != self.ignore_index) & (target < self.num_classes) & (pred < self.num_classes)
        
        # Filter pred and target using the valid mask
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

        # Update the confusion matrix (histogram)
        self.hist += np.bincount(
            self.num_classes * valid_target.flatten().astype(int) + valid_pred.flatten().astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def get_iou(self):
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iou

    def get_miou(self):
        iou = self.get_iou()
        miou = np.nanmean(iou)
        return miou

