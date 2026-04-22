import sys
sys.path.append('.')

import torch
from thop import profile
from module.pipe.U_Vit.u_vit import UViT

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 按照测试代码中的配置初始化 UViT模型
    # 注意：计算 FLOPs 时应禁用 use_checkpoint 以避免计算错误
    unet = UViT(
        img_size=64,
        patch_size=4,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=False, 
    ).to(device)
    
    unet.eval()

    # 构造 dummy inputs (Batch Size = 1)
    # 根据 UViT 的 forward 方法，通常输入为 (x, timesteps, y)
    dummy_x = torch.randn(1, 4, 64, 64).to(device)
    dummy_timesteps = torch.tensor([100.0]).to(device)
    dummy_y = torch.zeros(1, dtype=torch.long).to(device)

    # 运行 thop 进行计算
    with torch.no_grad():
        flops, params = profile(unet, inputs=(dummy_x, dummy_timesteps, dummy_y))
    
    print("=" * 30)
    print(f"Model: UViT")
    print(f"Parameters: {params / 1e6:.2f} M")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print("=" * 30)

if __name__ == "__main__":
    main()