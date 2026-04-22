# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange


# def build_palette(k=6, s=None):
#     """
#     构建调色板函数 - 直接替换版本
#     忽略k和s参数，使用预定义的手动调色板
#     保持相同的API接口但使用固定颜色
#     """
#     # 您提供的类别颜色
#     CLASS_COLORS = [
#         (0, 0, 0),          # Background - 黑色
#         (120, 120, 70),     # Shelf - 土黄色
#         (235, 235, 7),      # Jar or Tank - 亮黄色
#         (6, 230, 230),      # Freezer - 青蓝色
#         (204, 255, 4),      # Window - 黄绿色
#         (120, 120, 120),    # Glass Door - 中灰色
#         (140, 140, 140),    # Eyeglass - 浅灰色
#         (255, 51, 7),       # Cup - 红色
#         (224, 5, 255),      # Floor Glass - 洋红色
#         (204, 5, 255),      # Glass Bow - 紫色
#         (150, 5, 61),       # Water Bottle - 深红色
#         (4, 250, 7)         # Storage Box - 亮绿色
#     ]
    
#     # 创建PIL格式的调色板 (256种颜色 × 3通道 = 768长度)
#     palette = [0] * 768
    
#     # 将前12种颜色设置为指定的颜色
#     for i, color in enumerate(CLASS_COLORS):
#         palette[3*i] = color[0]    # R
#         palette[3*i+1] = color[1]  # G
#         palette[3*i+2] = color[2]  # B
    
#     return palette


# if __name__ == '__main__':
#     pass

###################################################################################

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange


# def build_palette(k=6,s=None):
#     if s==None:
#         s = 250 // (k-1)
#     else:
#         assert s*(k-1)<255
#     palette = []
#     for m0 in range(k):
#         for m1 in range(k):
#             for m2 in range(k):
#                 palette.extend([s*m0,s*m1,s*m2])
#     return palette




# if __name__ == '__main__':
#     pass



# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange


# def build_palette(k=6, s=None):
#     """
#     构建调色板函数 - 直接替换版本
#     忽略k和s参数，使用预定义的手动调色板
#     保持相同的API接口但使用固定颜色
#     """
#     # 您提供的类别颜色
#     CLASS_COLORS = [
#         (0, 0, 0),          # Background - 黑色
#         (120, 120, 70),     # Shelf - 土黄色
#         (235, 235, 7),      # Jar or Tank - 亮黄色
#         (6, 230, 230),      # Freezer - 青蓝色
#         (204, 255, 4),      # Window - 黄绿色
#         (120, 120, 120),    # Glass Door - 中灰色
#         (140, 140, 140),    # Eyeglass - 浅灰色
#         (255, 51, 7),       # Cup - 红色
#         (224, 5, 255),      # Floor Glass - 洋红色
#         (204, 5, 255),      # Glass Bow - 紫色
#         (150, 5, 61),       # Water Bottle - 深红色
#         (4, 250, 7)         # Storage Box - 亮绿色
#     ]
    
#     # 创建PIL格式的调色板 (256种颜色 × 3通道 = 768长度)
#     palette = [0] * 768
    
#     # 将前12种颜色设置为指定的颜色
#     for i, color in enumerate(CLASS_COLORS):
#         palette[3*i] = color[0]    # R
#         palette[3*i+1] = color[1]  # G
#         palette[3*i+2] = color[2]  # B
    
#     return palette


# if __name__ == '__main__':
#     pass


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange


# def build_palette(k=6,s=None):
#     if s==None:
#         s = 250 // (k-1)
#     else:
#         assert s*(k-1)<255
#     palette = []
#     for m0 in range(k):
#         for m1 in range(k):
#             for m2 in range(k):
#                 palette.extend([s*m0,s*m1,s*m2])
#     return palette


# if __name__ == '__main__':
#     pass





#####################################################################







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
    # 您提供的类别颜色
    palette = [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 128, 128, 128, 128, 0, 0, 0, 128, 0, 0, 0, 128]
    
    return palette


if __name__ == '__main__':
    pass




