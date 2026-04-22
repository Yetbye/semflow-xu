import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def build_palette(k=6, s=None):
    """
    构建调色板函数 - 直接替换版本
    忽略k和s参数，使用预定义的手动调色板
    保持相同的API接口但使用固定颜色
    颜色不再是原分割图的颜色，而是rgb空间最均匀的
    """
    
    palette = [0, 0, 0, 255, 255, 255]
    
    return palette