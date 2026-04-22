import sys
import os

# 将项目根目录（当前脚本的上一级目录）加入到包搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
from module.pipe.U_Vit.u_vit_rrdb import UViT

# ... 剩余代码保持不变 ...
def profile_model():
    # 检查是否有 GPU 可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 使用与你代码相同的参数初始化 UViT，作为推理测试将其 checkpoint 关闭
    model = UViT(
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
        use_checkpoint=False,  # 推理速度测试时禁用梯度检查点
    ).to(device)
    
    # 切换为测试模式
    model.eval()

    # 1. 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n================ 参数量统计 ================")
    print(f"总参数量:       {total_params / 1e6:.2f} M")
    print(f"可训练参数量:   {trainable_params / 1e6:.2f} M")

    # 2. 构造虚拟输入
    batch_size = 1  # 推理测试设为 batch_size = 1
    # 图像潜在特征 (latent feature) 输入尺寸 (bs, 4, 64, 64)
    dummy_x = torch.randn(batch_size, 4, 64, 64, device=device)
    # 随机 timesteps 
    dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    # 标签/条件图像 (RRDBNet包含缩放计算，根据其中的 spacial_dim=28，需要 224x224 输入)
    dummy_y = torch.randn(batch_size, 3, 224, 224, device=device)

    # 3. 预热 (Warm-up) 防止第一次执行由于显存分配导致时间偏大
    print(f"\n================ 速度测试 ==================")
    print("模型预热运行中 (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            model(dummy_x, dummy_timesteps, y=dummy_y)
    
    # 4. 评估推理性能
    num_iterations = 100
    print(f"正在测试推理速度 ({num_iterations} iterations)...")
    
    # 使用 CUDA Event 获取最准确的 GPU 运行时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        torch.cuda.synchronize()  # 确保所有之前的CUDA计算都已完成
        start_event.record()
        
        for _ in range(num_iterations):
            _x, _cls_logits = model(dummy_x, dummy_timesteps, y=dummy_y)
            
        end_event.record()
        torch.cuda.synchronize()  # 再次同步，等待循环中所有的测试完成
        
    # 计算平均时间
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iterations
    fps = 1000.0 / avg_time_ms
    
    print(f"平均每次推理时间: {avg_time_ms:.2f} ms")
    print(f"FPS (每秒推理帧数): {fps:.2f} 帧/秒")

if __name__ == "__main__":
    profile_model()