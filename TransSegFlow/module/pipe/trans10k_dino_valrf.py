import torch
import random
import os
import numpy as np
from torchvision.utils import save_image
from module.data.builder import build_palette
from module.metrics.iou import IoU
from module.pipe.pipe_uvit import pipeline_rf
from pathlib import Path
from tqdm import tqdm
from accelerate.logging import get_logger
from module.metrics.new_cls_evaluator import ClassificationEvaluator 
# --- Helper Functions from test.py ---

def rgb_to_class_id(rgb_images, palette, device, threshold=0.15):
    """Convert RGB images to class ID maps with a distance threshold."""
    palette = (palette.to(device).float() / 127.5) - 1.0
    palette = palette.view(1, -1, 3, 1, 1)  # (1, num_classes, 3, 1, 1)

    rgb_images = rgb_images.unsqueeze(1)
    dist = torch.sum((rgb_images - palette) ** 2, dim=2)
    min_dist, class_maps = torch.min(dist, dim=1)

    ignore_mask = min_dist > threshold
    class_maps[ignore_mask] = 255
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

def trans10k_dino_valrf(
    accelerator,
    args,
    vae,
    unet,
    processor,
    dino_model,
    val_dataloader,
    device,
    weight_dtype,
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
    total_correct = 0
    total_pixels = 0
    cls_evaluator = ClassificationEvaluator()
    # 3. Setup Timesteps (Same as test.py)
    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    
    # 使用demo.py/test.py中的timesteps处理方式
    timesteps = torch.arange(1, 1000, 1000 // num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)
    
    palette = torch.tensor(build_palette(args.pa.k, args.pa.s), dtype=torch.uint8)
    
    # 4. Loop
    progress_bar = tqdm(val_dataloader, disable=not accelerator.is_local_main_process, desc=f"Validation Step {gstep}")
    

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
             # 维持你之前的这步
            raw_images = (batch['image'].to(device=device, dtype=weight_dtype) + 1.0) / 2.0

            inputs = processor(images=raw_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=weight_dtype)

            outputs = dino_model(pixel_values=pixel_values)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                dino_feat = outputs.pooler_output
            else:
                dino_feat = outputs.last_hidden_state[:, 0]
            # Predict
            pred_latents, _,all_cls_preds = pipeline_rf(
                timesteps, 
                unet, 
                image_latents, 
                guidance_scale, 
                images,
                dino_feat,
                None
            )
            final_cls_logits = all_cls_preds[-1] 
           
            cls_targets = batch['cls_target'].to(device=device, dtype=weight_dtype) 
            cls_evaluator.update(final_cls_logits, cls_targets)
            pred_rgb = vae.decode(pred_latents / vae.config.scaling_factor).sample
            
            # Convert to Class IDs
            pred_classes = rgb_to_class_id(pred_rgb, palette, device)
            
            # Load Target
            target_classes = rgb_to_class_id(batch['image_semseg'].to(device=device, dtype=weight_dtype), palette, device)
            
            # Update Metric
            pred_classes_np = pred_classes.cpu().numpy()
            target_np = target_classes.cpu().numpy()
            iou_metric.add(pred_classes_np, target_np)
            # --- 新增 Pixel Accuracy 计算 ---
            ignore_index = real_dataset.ignore_index
            valid_mask = target_np != ignore_index
            total_correct += np.sum((pred_classes_np == target_np) & valid_mask)
            total_pixels += np.sum(valid_mask)

            # Visualization (Randomly save)
            if i % 10 == 0: # Save every 10th batch
                 # Pick first image in batch
                idx = 0 
                
                # Custom color map for visualization (from test.py)
                CLASS_COLORS = [
                (0, 0, 0), (120, 120, 70), (235, 235, 7),
                (6, 230, 230), (204, 255, 4), (120, 120, 120),
                (140, 140, 140), (255, 51, 7), (224, 5, 255),
                (204, 5, 255), (150, 5, 61), (4, 250, 7)
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
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    # 获取分类准确率
    cls_acc = cls_evaluator.compute()
    # 打印到日志
    logger.info(f"\n[Validation {gstep}] mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f} | Cls Acc: {cls_acc:.4f}")
    
    # 仅主进程写入文件
    if accelerator.is_local_main_process:
        print(f"Validation {gstep} mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f} | Cls Acc: {cls_acc:.4f}")
        
        val_txt_path = os.path.join(args.env.output_dir, "val.txt")
        with open(val_txt_path, "a") as f:
            f.write(f"\n{'='*20} Step {gstep} {'='*20}\n")
            f.write(f"mIoU: {miou:.4f}\n")
            f.write(f"Pixel Acc: {pixel_acc:.4f}\n")
            f.write(f"Cls Acc: {cls_acc:.4f}\n") 
            for k, class_iou in enumerate(iou):
                cls_info = f"Class {k} IoU: {class_iou:.4f}"
                print(cls_info)
                logger.info(cls_info)
                f.write(cls_info + "\n")
            f.write(f"{'='*50}\n")

    # Log to tracking (wandb/tensorboard)
    accelerator.log({
        "val/mIoU": miou, 
        "val/pixel_acc": pixel_acc,
        "val/cls_acc": cls_acc
    }, step=gstep)
    
    return miou