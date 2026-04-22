import torch
import random
import os
from torchvision.utils import save_image
from module.data.builder import build_palette
from module.metrics.iou import IoU
from module.pipe.pipe_unet import pipeline_rf
from pathlib import Path
from tqdm import tqdm
from accelerate.logging import get_logger
# --- Helper Functions from test.py ---

def rgb_to_class_id(rgb_images, palette, device, threshold=0.15):
    """Convert RGB images to class ID maps with a distance threshold."""
    palette = (palette.to(device).float() / 127.5) - 1.0
    palette = palette.view(1, -1, 3, 1, 1)  # (1, num_classes, 3, 1, 1)

    rgb_images = rgb_images.unsqueeze(1)
    dist = torch.sum((rgb_images - palette) ** 2, dim=2)
    min_dist, class_maps = torch.min(dist, dim=1)

    ignore_mask = min_dist > threshold
    class_maps[ignore_mask] = 0
    return class_maps

def class_id_to_rgb(class_ids, palette):
    """Convert class ID maps to RGB images."""
    if not isinstance(palette, torch.Tensor):
        palette = torch.tensor(palette, dtype=torch.uint8)
    palette = palette.to(class_ids.device)
    if palette.dim() == 1:
        palette = palette.view(-1, 3)

    rgb_images = torch.zeros((class_ids.shape[0], 3, class_ids.shape[1], class_ids.shape[2]), dtype=torch.uint8, device=class_ids.device)

    for class_id in range(palette.shape[0]):
        mask = class_ids == class_id
        rgb_images[:, 0, :, :][mask] = palette[class_id, 0]
        rgb_images[:, 1, :, :][mask] = palette[class_id, 1]
        rgb_images[:, 2, :, :][mask] = palette[class_id, 2]

    ignore_mask = class_ids == 255
    rgb_images[:, 0, :, :][ignore_mask] = 0
    rgb_images[:, 1, :, :][ignore_mask] = 0
    rgb_images[:, 2, :, :][ignore_mask] = 0

    return rgb_images.float() / 255.0

# --- Main Function ---

def unet_gdd_val(
    accelerator,
    args,
    vae,
    unet,
    val_dataloader,
    device,
    weight_dtype,
    null_condition,
    max_iter=None,
    gstep=0
):
    """
    Validation function similar to test.py logic (segmentation IoU).
    """
    logger = get_logger(__name__)  # 或者使用 logging.getLogger(__name__)
    
    # 1. Setup Directories
    vis_dir = Path(args.env.output_dir) / "vis" / f"step_{gstep}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Metrics
    dataset = val_dataloader.dataset
    if hasattr(dataset, 'dataset'): # Handle Subset or other wrappers
        real_dataset = dataset.dataset
    else:
        real_dataset = dataset
        
    iou_metric = IoU(num_classes=real_dataset.num_classes, ignore_index=real_dataset.ignore_index)
    # [新增] 初始化二进制混淆矩阵的统计量，用于 F_beta 和 BER
    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0
    
    # [新增] 假设你的前景(目标)类别 ID 是 1
    fg_class_id = 1
    # 3. Setup Timesteps (Same as test.py)
    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    
    # 使用demo.py/test.py中的timesteps处理方式
    timesteps = torch.arange(1, 1000, 1000 // num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)
    
    palette = torch.tensor(build_palette(args.pa.k, args.pa.s), dtype=torch.uint8)
    
    # 4. Loop
    progress_bar = tqdm(val_dataloader, disable=not accelerator.is_local_main_process, desc=f"Validation Step {gstep}")
    
    # 缓存 prompt embeds
    prompt_embeds = null_condition

    for i, batch in enumerate(progress_bar):
        if max_iter is not None and i >= max_iter:
            break
            
        # [Fix] Handle None batch from collate_fn failure
        if batch is None:
            continue
            
        images = batch['image'].to(dtype=weight_dtype, device=device)
        
        # 融入噪声 (Test-Time Augmentation logic from test.py)
        
        images_input = images 
        
        bsz = images.shape[0]
        
        with torch.no_grad():
            image_latents = vae.encode(images_input).latent_dist.mode() * vae.config.scaling_factor
            
            encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)
            
            # Predict
            pred_latents, _ = pipeline_rf(
                timesteps, 
                unet, 
                image_latents, 
                encoder_hidden_states, 
                prompt_embeds,
                guidance_scale, 
                None
            )
            
            pred_rgb = vae.decode(pred_latents / vae.config.scaling_factor).sample
            
            # Convert to Class IDs
            pred_classes = rgb_to_class_id(pred_rgb, palette, device)
            
            # Load Target
            target_classes = rgb_to_class_id(batch['image_semseg'].to(device=device, dtype=weight_dtype), palette, device)
            
            # Update Metric
            pred_classes_np = pred_classes.cpu().numpy()
            target_np = target_classes.cpu().numpy()
            iou_metric.add(pred_classes_np, target_np)
            # [新增] 计算当前批次的 TP, FP, TN, FN
            # 过滤掉 ignore_index (通常是 255)
            valid_mask = (target_np != real_dataset.ignore_index)
            
            # 提取预测和真实标签中的前景(目标)部分
            pred_fg = (pred_classes_np == fg_class_id)[valid_mask]
            target_fg = (target_np == fg_class_id)[valid_mask]
            
            # 累加统计值 (布尔运算计算交集)
            total_tp += (pred_fg & target_fg).sum()
            total_fp += (pred_fg & ~target_fg).sum()
            total_tn += (~pred_fg & ~target_fg).sum()
            total_fn += (~pred_fg & target_fg).sum()

            # Visualization (Randomly save)
            if i % 10 == 0: # Save every 10th batch
                 # Pick first image in batch
                idx = 0 
                
                # Custom color map for visualization (from test.py)
                CLASS_COLORS = [
                    (0, 0, 0),   # road
                    (255, 255, 255),   # sidewalk
                ]

                save_palette = torch.tensor(CLASS_COLORS, dtype=torch.uint8)
                
                pred_vis = class_id_to_rgb(pred_classes[idx:idx + 1], save_palette)
                target_vis = class_id_to_rgb(target_classes[idx:idx + 1], save_palette)
                
                save_image(pred_vis, vis_dir / f"batch_{i}_pred.png")
                save_image(target_vis, vis_dir / f"batch_{i}_gt.png")
                # Save Raw Prediction RGB for debugging
                save_image(pred_rgb[idx:idx+1], vis_dir / f"batch_{i}_pred_raw_rgb.png")

    # 5. Summary
    miou = iou_metric.get_miou()
    iou = iou_metric.get_iou()
    # [新增] 计算 F_beta 和 BER
    epsilon = 1e-7 # 防止除以0
    # [新增] 计算 Pixel Accuracy
    total_pixels = total_tp + total_tn + total_fp + total_fn + epsilon
    pixel_acc = (total_tp + total_tn) / total_pixels
    # Precision 和 Recall
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    
    # 计算 F_beta (按照你的要求，设置 beta_sq = 0.3)
    beta_sq = 0.3
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall + epsilon)
    
    # 计算 BER
    fpr = total_fp / (total_tn + total_fp + epsilon) # False Positive Rate
    fnr = total_fn / (total_tp + total_fn + epsilon) # False Negative Rate
    ber = (fpr + fnr) / 2.0
    # 打印到日志
    # 打印到日志
    logger.info(f"\n[Validation {gstep}] mIoU: {miou:.4f} | Pixel_Acc: {pixel_acc:.4f} | F-beta(0.3): {f_beta:.4f} | BER: {ber:.4f}")
    
    # 仅主进程写入文件
    if accelerator.is_local_main_process:
        print(f"Validation {gstep} mIoU: {miou:.4f} | Pixel_Acc: {pixel_acc:.4f} | F-beta(0.3): {f_beta:.4f} | BER: {ber:.4f}")
        
        val_txt_path = os.path.join(args.env.output_dir, "val.txt")
        with open(val_txt_path, "a") as f:
            f.write(f"\n{'='*20} Step {gstep} {'='*20}\n")
            f.write(f"mIoU: {miou:.4f}\n")
            f.write(f"Pixel_Acc: {pixel_acc:.4f}\n")
            f.write(f"F-beta(0.3): {f_beta:.4f}\n")
            f.write(f"BER: {ber:.4f}\n")
            for k, class_iou in enumerate(iou):
                cls_info = f"Class {k} IoU: {class_iou:.4f}"
                print(cls_info)
                logger.info(cls_info)
                f.write(cls_info + "\n")
            f.write(f"{'='*50}\n")

    # Log to tracking (wandb/tensorboard)
    accelerator.log({"val/mIoU": miou,
                        "val/Pixel_Acc": pixel_acc,
                     "val/F_beta_0.3": f_beta,
                     "val/BER": ber}, step=gstep)
    
    return miou