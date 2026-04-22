# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimplifiedASL(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=1, eps=1e-8):
#         super().__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.eps = eps

#     def forward(self, logits, targets):
#         # logits: 模型输出
#         # targets: 真实标签 (y)
        
#         # p_k
#         probs = torch.sigmoid(logits)
        
#         # -------------------------------------------------------
#         # 对应公式第一行: y_k = 1 的情况
#         # 公式: (1 - p_k)^(gamma+) * log(p_k)
#         # -------------------------------------------------------
#         # targets 就是 y_k。当 y_k=1 时保留，y_k=0 时这一项为0
#         pos_term = targets * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs + self.eps)
        
#         # -------------------------------------------------------
#         # 对应公式第二行: y_k = 0 的情况
#         # 公式: (p_k)^(gamma-) * log(1 - p_k)
#         # -------------------------------------------------------
#         # (1 - targets) 就是判断 y_k=0。
#         neg_term = (1 - targets) * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs + self.eps)
        
#         # -------------------------------------------------------
#         # 对应公式最前面的求和符号 ∑ 和系数 1/K
#         # -------------------------------------------------------
#         # 这里的负号 '-' 是为了把公式里的负值变成正 Loss
#         loss = - (pos_term + neg_term)
        
#         # mean() 就等于公式里的 1/K * ∑
#         return loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedASL(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=1, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型直接输出的 Logits (未经过 Sigmoid)
            targets: 真实标签 (0 或 1)
        """
        # 1. 计算概率 p (仅用于权重项，不参与 log 计算)
        # 使用 sigmoid 得到概率
        probs = torch.sigmoid(logits)
        
        # 2. 计算 LogSigmoid (核心修复点)
        # log(p) = F.logsigmoid(logits)
        # log(1-p) = F.logsigmoid(-logits)
        # 这种写法在 logits 很大或很小时都能保持梯度稳定，不会出现 NaN
        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)

        # 3. 计算正样本项 (y=1)
        # Loss+ = - y * (1-p)^gamma+ * log(p)
        pos_weight = torch.pow(1 - probs, self.gamma_pos)
        pos_loss = targets * pos_weight * log_p
        
        # 4. 计算负样本项 (y=0)
        # Loss- = - (1-y) * p^gamma- * log(1-p)
        neg_weight = torch.pow(probs, self.gamma_neg)
        neg_loss = (1 - targets) * neg_weight * log_not_p
        
        # 5. 合并
        loss = - (pos_loss + neg_loss)
        
        return loss.mean()