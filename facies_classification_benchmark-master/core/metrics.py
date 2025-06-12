import numpy as np

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        
        # 检查混淆矩阵是否有效
        if hist.sum() == 0:
            # 如果混淆矩阵全为零，返回所有零指标
            return {
                'Pixel Acc': 0.0,
                'Class Accuracy': 0.0,
                'Mean Class Acc': 0.0,
                'Freq Weighted IoU': 0.0,
                'Mean IoU': 0.0,
                'confusion_matrix': hist
            }, {i: 0.0 for i in range(self.n_classes)}
            
        # 安全计算总体精度
        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        
        # 安全计算每个类别的精度
        # 添加小的常数避免除零
        sum_axis1 = hist.sum(axis=1)
        acc_cls = np.zeros_like(sum_axis1, dtype=np.float64)
        valid_indices = sum_axis1 > 0
        acc_cls[valid_indices] = np.diag(hist)[valid_indices] / (sum_axis1[valid_indices] + 1e-10)
        acc_cls = np.nanmean(acc_cls)
        
        # 计算每个类别的准确率
        cls_acc = np.zeros_like(sum_axis1, dtype=np.float64)
        cls_acc[valid_indices] = np.diag(hist)[valid_indices] / (sum_axis1[valid_indices] + 1e-10)
        mean_cls_acc = np.nanmean(cls_acc)
        
        # 安全计算IoU
        denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = np.zeros_like(denominator, dtype=np.float64)
        valid_indices = denominator > 0
        iu[valid_indices] = np.diag(hist)[valid_indices] / (denominator[valid_indices] + 1e-10)
        mean_iu = np.nanmean(iu)
        
        # 安全计算频率加权IoU
        freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
        valid_indices = (freq > 0) & (iu >= 0)  # 过滤掉nan和inf
        fwavacc = (freq[valid_indices] * iu[valid_indices]).sum()
        
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            'Pixel Acc': float(acc),  # 确保返回Python标量
            'Class Accuracy': float(acc_cls),
            'Mean Class Acc': float(mean_cls_acc),
            'Freq Weighted IoU': float(fwavacc),
            'Mean IoU': float(mean_iu),
            'confusion_matrix': hist
        }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
