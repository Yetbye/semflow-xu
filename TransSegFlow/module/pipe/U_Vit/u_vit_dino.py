# import torch
# import torch.nn as nn
# import math
# from .timm import trunc_normal_, Mlp
# import einops
# import torch.utils.checkpoint

# if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
#     ATTENTION_MODE = 'flash'
# else:
#     try:
#         import xformers
#         import xformers.ops
#         ATTENTION_MODE = 'xformers'
#     except:
#         ATTENTION_MODE = 'math'
# print(f'attention mode is {ATTENTION_MODE}')


# def timestep_embedding(timesteps, dim, max_period=10000):
#     """
#     Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding


# def patchify(imgs, patch_size):
#     x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
#     return x


# def unpatchify(x, channels=3):
#     patch_size = int((x.shape[2] // channels) ** 0.5)
#     h = w = int(x.shape[1] ** .5)
#     assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
#     x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
#     return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, L, C = x.shape

#         qkv = self.qkv(x)
#         if ATTENTION_MODE == 'flash':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
#             x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#             x = einops.rearrange(x, 'B H L D -> B L (H D)')
#         elif ATTENTION_MODE == 'xformers':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
#             x = xformers.ops.memory_efficient_attention(q, k, v)
#             x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
#         elif ATTENTION_MODE == 'math':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = (attn @ v).transpose(1, 2).reshape(B, L, C)
#         else:
#             raise NotImplemented

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
#         self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
#         self.use_checkpoint = use_checkpoint

#     def forward(self, x, skip=None):
#         if self.use_checkpoint:
#             return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
#         else:
#             return self._forward(x, skip)

#     def _forward(self, x, skip=None):
#         if self.skip_linear is not None:
#             x = self.skip_linear(torch.cat([x, skip], dim=-1))
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, patch_size, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H % self.patch_size == 0 and W % self.patch_size == 0
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x


# class UViT(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
#                  qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
#                  use_checkpoint=False, conv=True, skip=True):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_classes = num_classes
#         self.in_chans = in_chans

#         self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = (img_size // patch_size) ** 2

#         self.time_embed = nn.Sequential(
#             nn.Linear(embed_dim, 4 * embed_dim),
#             nn.SiLU(),
#             nn.Linear(4 * embed_dim, embed_dim),
#         ) if mlp_time_embed else nn.Identity()

#         if self.num_classes > 0:
#             self.label_emb = nn.Embedding(self.num_classes, embed_dim)
#             self.extras = 2
#         else:
#             self.extras = 1

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

#         self.in_blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 norm_layer=norm_layer, use_checkpoint=use_checkpoint)
#             for _ in range(depth // 2)])

#         self.mid_block = Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 norm_layer=norm_layer, use_checkpoint=use_checkpoint)

#         self.out_blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
#             for _ in range(depth // 2)])

#         self.norm = norm_layer(embed_dim)
#         self.patch_dim = patch_size ** 2 * in_chans
#         self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
#         self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed'}

#     def forward(self, x, timesteps, y=None):
#         x = self.patch_embed(x)
#         B, L, D = x.shape

#         time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
#         time_token = time_token.unsqueeze(dim=1)
#         x = torch.cat((time_token, x), dim=1)
#         if y is not None:
#             label_emb = self.label_emb(y)
#             label_emb = label_emb.unsqueeze(dim=1)
#             x = torch.cat((label_emb, x), dim=1)
#         x = x + self.pos_embed

#         skips = []
#         for blk in self.in_blocks:
#             x = blk(x)
#             skips.append(x)

#         x = self.mid_block(x)

#         for blk in self.out_blocks:
#             x = blk(x, skips.pop())

#         x = self.norm(x)
#         x = self.decoder_pred(x)
#         assert x.size(1) == self.extras + L
#         x = x[:, self.extras:, :]
#         x = unpatchify(x, self.in_chans)
#         x = self.final_layer(x)
#         return x

# 导入 PyTorch 相关库
import torch
import torch.nn as nn
import math
# 从同一个目录中的 timm 模块导入工具
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint

# 检测当前环境支持的注意力机制计算模式
# 优先使用 PyTorch 2.0 内置的 scaled_dot_product_attention (支持 flash attention)
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        # 如果没有内置，尝试使用 xformers 库以获得更快的自注意力
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        # 如果两者都没有，则退回使用基础的数学实现
        ATTENTION_MODE = 'math'
# 打印当前使用的注意力模式
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建基于正弦和余弦的时间步嵌入 (类似 Transformer 里的位置编码)。
    """
    # 频率数量为嵌入维度的一半
    half = dim // 2
    # 计算各个维度的频率因子
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # 将时间步与频率相乘，得出对应角度
    args = timesteps[:, None].float() * freqs[None]
    # 将余弦和正弦特征拼接，产生完整的嵌入表示
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 如果维度是奇数，在末尾补零使得维度对齐
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    # 返回时间步嵌入张量 (N, dim)
    return embedding


def patchify(imgs, patch_size):
    # 使用 einops 库将 2D 图像划分成一系列的展平的 patch
    # 形状转换规则：从 B C H W 转换为 B (h*w) (patch_H * patch_W * C)
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    # 根据最后一维的大小和通道数推算出 patch 的大小
    patch_size = int((x.shape[2] // channels) ** 0.5)
    # 推算原始图像特征图的高和宽 (在 patch 级别)
    h = w = int(x.shape[1] ** .5)
    # 检查空间尺寸重构是否合法
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    # 将展平的 patch 重新拼接回 2D 图像特征图的形式
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    # 注意力模块的定义
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 记录注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头的维度大小
        head_dim = dim // num_heads
        # 记录缩放系数，如果没有显式传入，则使用头维度的开方分之一
        self.scale = qk_scale or head_dim ** -0.5

        # 定义产生 Query, Key, Value 的线性投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力得分的 Dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        # 注意力计算后的输出映射层
        self.proj = nn.Linear(dim, dim)
        # 输出的 Dropout 层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 获取输入的形状：Batch, 序列长度, 维度
        B, L, C = x.shape

        # 一次性计算得到所有 Query, Key, Value 拼接的特征
        qkv = self.qkv(x)
        # 根据系统支持的注意力模式选择高效实现方式
        if ATTENTION_MODE == 'flash':
            # 重排形状为 K, Batch, 头数, 序列长, 头的维度以适配 flash_attention
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            # 调用 PyTorch 内置的加速 dot product attention
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # 再融合不同头得到最终特征特征
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            # 适配 xformers 的输入形状规范
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            # 使用 xformers 提供的内存高效计算
            x = xformers.ops.memory_efficient_attention(q, k, v)
            # 重排恢复输出形状
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            # 基础数学方式计算自注意力 (内存使用率高)
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            # QK^T 并随缩放因子缩放
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # 使用 Softmax 归一化得分
            attn = attn.softmax(dim=-1)
            # 加上注意力 DropOut
            attn = self.attn_drop(attn)
            # 注意力权重乘上 Value 项，最后转置和 reshape 获取对应尺寸
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented # 抛出未实现异常

        # 执行最后的线性映射
        x = self.proj(x)
        # 经过输出层 dropout
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # 定义标准 Transformer Block (附加可能的 skip connection)
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        # 定义进入自注意力之前的归一化层
        self.norm1 = norm_layer(dim)
        # 定义自注意力层
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        # 定义进入 MLP 之前的归一化层
        self.norm2 = norm_layer(dim)
        # 根据 mlp_ratio 计算 MLP 内部的隐藏维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 定义 MLP 模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        # 如果需要引入跳跃连接 (skip)，则声明一个调整维度的层
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        # 是否在此块内启用梯度检查点 (用于节省显存)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        # 通过 checkpoint 判断是否直接执行前向传播
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        # 实际的 Block 逻辑计算
        if self.skip_linear is not None:
            # 如果存在同层传递过来的 skip connection 特征，将其拼接并且通过线性层融合成基础维度
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        # 执行带有残差的归一化与自注意力
        x = x + self.attn(self.norm1(x))
        # 执行带有残差的归一化与多层感知机 (MLP)
        x = x + self.mlp(self.norm2(x))
        # 返回块的输出
        return x


class PatchEmbed(nn.Module):
    """ 将图像映射为 Patch 的嵌入 (Image to Patch Embedding)
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 利用二维卷积同时实现划窗和特征映射 (stride=patch_size实现互不交叠的块)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 断言原始图像宽、高能够被 patch 大小整除
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        # 从 2D 卷积的输出 (B, D, H', W') 展平并调整维度顺序，输出为 (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    # 定义基于 U-Net 思想结合 Vision Transformer (ViT) 的主干网络 UViT 架构
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        # 记录特征的维度
        self.num_features = self.embed_dim = embed_dim
        # 类别的总数 (用于类别条件嵌入)
        self.num_classes = num_classes
        # 输入通道数量
        self.in_chans = in_chans

        # 实例化 Patch 嵌入模块
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # 计算整张图像有多少个 Patch
        num_patches = (img_size // patch_size) ** 2

        # 构造用于时间步条件 (Diffusion models的time step) 的多层嵌入网络
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), # 第一层全连接，保持维度
            nn.Tanh(),                           # 激活函数 (BERT用Tanh, ViT常用GELU, 这里可以用 Tanh 或 nn.GELU())
            nn.Linear(embed_dim, 11) # 第二层全连接，输出类别
        )

        # 根据是否是有条件分类模型来初始化类别嵌入，并确认位置嵌入中的预留 token 数量
        if self.num_classes > 0:
            dino_dim = 384
            self.label_emb = nn.Sequential(
                nn.LayerNorm(dino_dim),
                nn.Linear(dino_dim, embed_dim),
                nn.SiLU(),  # 激活函数，也可以用 nn.GELU()
                nn.Linear(embed_dim, embed_dim)
            )
            self.extras = 2 # 如果有类别需要额外预留2个token (时间步、类别)
        else:
            self.extras = 1 # 如果没有类别只需要额外预留1个token (时间步)

        # 定义可学习的全体 Patch 的位置编码参数，包括额外的 Token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        # 定义下采样部分 (Encoder) 的多个 Transformer Block (深度的一半)
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        # 定义位于模型中间 (Bottleneck) 的一个 Transformer Block
        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        # 定义上采样部分 (Decoder) 的多个 Transformer Block，这些 Block 会开启跳跃连接接收 Encoder 的特征
        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        # 最后的全局归一化层
        self.norm = norm_layer(embed_dim)
        # 初始化 patch 对应到像素原图所需的恢复维度
        self.patch_dim = patch_size ** 2 * in_chans
        # 使用线性层来将通道解码回原始图像维数
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        # 最后可能附加的一个卷积层用于平滑重构后的像素
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        # 使用截断正态分布初始化位置嵌入
        trunc_normal_(self.pos_embed, std=.02)
        # 遍历所有子模块，运行模型权重初始化策略
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 权重初始化函数，针对不同模块(全连接层和层归一化)按规范设定方差和零偏置
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # 指定在使用优化器时不需要进行权值衰减的变量，这里指屏蔽位置编码
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        # 步骤 1：从图片提取 Patch Embedding
        x = self.patch_embed(x)
        # 取出其批大小(B)、序列长度(L)、嵌入维数(D)
        B, L, D = x.shape

        # 步骤 2：对当前的时间步进行特征映射提取，添加为序列的前置 token 之一
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        # 步骤 3：如果包含类别标签 y，也映射为标签嵌入，作为额外前置 token 加入序列
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)

        # 步骤 4：添加所有 Patch (和 Token) 的全局位置编码 (广播相加)
        x = x + self.pos_embed

        # 存储 Encoder 端产生的特征用于跳跃连接
        skips = []
        for blk in self.in_blocks:
            # 依次经过下采样段的所有 Block
            x = blk(x)
            # 压入列表进行缓存
            skips.append(x)

        # 经过处于底部的中间 Block (不缓存 skip)
        x = self.mid_block(x)

        for blk in self.out_blocks:
            # 依次经过上采样段的 Block，通过 pop() 获取对应的对应 Encoder特征实现 U 型跳跃连接
            x = blk(x, skips.pop())

        # 获取输出后通过 LayerNorm
        x = self.norm(x)

        # --- 新增的分类逻辑开始 ---
        # 1. 过滤掉前置的特殊 Token (时间步、类别)，只保留图像 Patch 特征
        patch_features = x[:, self.extras:, :] # 形状: (B, L, D)
        # 2. 对所有 Patch 特征在序列维度（dim=1）进行平均池化，得到全局图像特征
        global_features = patch_features.mean(dim=1) # 形状: (B, D)
        # 3. 送入分类头，输出整图的类别 logits
        cls_logits = self.cls_head(global_features)  # 形状: (B, 11)
        # --- 新增的分类逻辑结束 ---

        # 将 Transformer 维度解码回 Patch 的像素展开维度
        x = self.decoder_pred(x)
        
        # 验证经过一系列操作后长度仍合理
        assert x.size(1) == self.extras + L
        # 舍弃序列前面的条件 tokens（时间步、类别），仅截取与 Patch 对应的特征序列
        x = x[:, self.extras:, :]
        # 解开 Patch，恢复为图像特征图的形状
        x = unpatchify(x, self.in_chans)
        # 执行最后的输出卷积平滑
        x = self.final_layer(x)
        return x, cls_logits
        # 返回最终网络预测结果