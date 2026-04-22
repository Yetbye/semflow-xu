"""
Heavily modified from REPA: https://github.com/sihyun-yu/REPA/blob/main/loss.py
"""
# 从 REPA 项目大量修改而来：https://github.com/sihyun-yu/REPA/blob/main/loss.py

import pdb # 导入 Python 调试器
import torch # 导入 PyTorch 库
import numpy as np # 导入 NumPy 库，用于数值运算
import torch.nn.functional as F # 导入 PyTorch 的神经网络函数


def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    对所有非批次维度取均值。
    """
    # 计算张量 x 在除了第0维（批次维）之外的所有维度上的均值
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x, temperature=1.):
    """
    Take the mean over all non-batch dimensions.
    对所有非批次维度取和。
    """
    # 计算张量 x 在除了第0维（批次维）之外的所有维度上的和
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def mse_flat(x, y, temperature=1, **kwargs):
    # 计算 x 和 y 之间的均方误差（MSE）
    err = (x - y) ** 2 # 计算 (x - y) 的平方
    return mean_flat(err) # 对误差在非批次维度上取均值


def class_conditioned_sampling(labels):
    """
    根据类别标签进行负采样 (支持 Multi-Hot Labels)。
    
    Args:
        labels: (Batch_Size, Num_Classes) 的 Multi-Hot 张量 (0或1)
    """
    bsz = labels.shape[0]
    
    # 【关键修改】计算两两样本之间的类别重叠
    # labels: (B, C)
    # labels.t(): (C, B)
    # intersection: (B, B)
    # intersection[i, j] 表示样本 i 和样本 j 共同拥有的类别数量
    intersection = torch.mm(labels, labels.t())
    
    # 定义负样本掩码：
    # 如果 intersection[i, j] == 0，说明两个样本没有任何共同类别 -> 可以作为负样本 (True)
    # 如果 intersection[i, j] > 0，说明有共同类别 (或者是自己) -> 不能作为负样本 (False)
    mask = (intersection == 0)
    
    # --- 以下逻辑与之前相同 ---
    
    # 计算每个样本有多少个合法的负样本候选
    weights = mask.float() 
    weights_sum = weights.sum(dim=1, keepdim=True) 
    
    # 标记哪些样本有有效的负样本
    valid_mask = (weights_sum > 0).squeeze(1) 
    
    # 初始化 choices
    choices = torch.zeros(bsz, dtype=torch.long, device=labels.device)
    
    # 只对有有效负样本的行进行采样
    if valid_mask.any():
        valid_weights = weights[valid_mask]
        valid_weights_sum = weights_sum[valid_mask]
        
        # 归一化权重
        valid_weights = valid_weights / valid_weights_sum.clamp(min=1e-6)
        
        # 采样
        valid_choices = torch.multinomial(valid_weights, 1).squeeze(1)
        
        # 填回
        choices[valid_mask] = valid_choices

    return choices, valid_mask

def compute_class_conditioned_triplet_loss(x, y, labels, temperature=1.0):
    """
    计算基于类别的三元组损失。
    如果某个样本在 Batch 中找不到异类负样本，则该样本只计算正样本损失，不计算负样本损失。
    """
    # 1. 展平特征
    x = x.flatten(1) 
    y = y.flatten(1)
    
    # 2. 计算正样本误差 (Positive Loss) - 所有样本都计算
    pos_error = mean_flat((x - y) ** 2)

    # 3. 获取负样本索引和有效性掩码
    neg_indices, valid_mask = class_conditioned_sampling(labels)
    
    # 4. 计算负样本误差 (Negative Loss)
    # 初始化负样本误差为 0
    neg_error = torch.zeros_like(pos_error)
    
    if valid_mask.any():
        # 只提取有效的负样本特征
        # 注意：这里我们只取 valid_mask 为 True 的部分进行计算，节省计算量
        x_valid = x[valid_mask]
        y_neg_valid = y[neg_indices[valid_mask]]
        
        # 计算这些有效样本的距离
        valid_neg_error = mean_flat((x_valid - y_neg_valid) ** 2)
        
        # 将计算结果填回 neg_error 对应的位置
        neg_error[valid_mask] = valid_neg_error

    # 5. 计算总损失
    # 对于 valid_mask 为 False 的样本，neg_error 是 0，所以 Loss = pos_error - 0
    # 对于 valid_mask 为 True 的样本，Loss = pos_error - temp * neg_error
    loss = pos_error - temperature * neg_error

    return {
        "loss": loss,
        "flow_loss": pos_error,
        "contrastive_loss": neg_error
    }

# def compute_triplet_loss_efficiently( x, y,labels, temperature=1.0):
#     # 高效计算三元组损失（随机负采样）
#     x = x.flatten(1) # 将预测 x 展平为 (batch_size, -1)
#     y = y.flatten(1) # 将目标 y 展平为 (batch_size, -1)
#     # 获取正样本并计算误差
#     y_pos = y # 正样本就是目标 y 本身
#     pos_error = mean_flat((x - y_pos) ** 2) # 计算正样本对的均方误差
#     bsz = x.shape[0] # 获取批次大小
#     # 创建一个索引矩阵用于选择负样本
#     choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(x.device)
#     choices.fill_diagonal_(-1.) # 将对角线填充为-1，以排除自身
#     choices = choices.sort(dim=1)[0][:, 1:] # 排序并移除-1，得到除自身外的所有索引
#     # 为每个样本随机选择一个负样本
#     choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
#     y_neg = y[choices] # 根据索引选择负样本
#     # 计算误差
#     non_nulls = torch.ones(bsz, dtype=torch.bool, device=x.device) # 否则，所有样本都参与对比
#     # 计算负样本对的元素级误差，并应用掩码
#     neg_elem_error = ((x - y_neg) ** 2) * non_nulls.to(x.device).unsqueeze(-1)
#     neg_elem_error = neg_elem_error # 这行代码没有实际作用
#     # 计算负样本误差的均值，并根据参与对比的样本数量进行缩放
#     neg_error = mean_flat(neg_elem_error) * bsz / non_nulls.sum() # 重新缩放以考虑空类别
#     # 计算最终损失
#     loss = pos_error - temperature * neg_error # 损失 = 正样本误差 - 温度 * 负样本误差
#     # return loss
#     return { # 返回一个包含总损失、流损失（正样本误差）和对比损失（负样本误差）的字典
#         "loss": loss,
#         "flow_loss": pos_error,
#         "contrastive_loss": neg_error
#     }


