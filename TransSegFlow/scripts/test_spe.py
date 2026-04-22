import sys
import os
from pathlib import Path

sys.path.append('.')

import logging
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm import tqdm
import random
from torchvision.utils import save_image
from safetensors.torch import load_file
from module.data.load_dataset import pr_val_dataloader
from module.pipe.pipe import pipeline_rf_reverse
from module.data.builder import build_palette
from einops import rearrange
from module.metrics.iou import IoU

logger = get_logger(__name__)

def get_unet_added_conditions(args, null_condition):
    prompt_embeds = null_condition
    unet_added_conditions = None
    return prompt_embeds, unet_added_conditions

def rgb_to_class_id(rgb_images, palette, device, threshold=1.55):
    # 使用更精确的转换公式，确保与图像的[-1, 1]范围一致
    # 阈值设置为1.55，基于颜色距离分析，以在避免ignore_index分配和防止类别混淆之间取得平衡
    palette_normalized = (palette.to(device).float() / 255.0) * 2.0 - 1.0
    palette_normalized = palette_normalized.view(1, -1, 3, 1, 1)
    rgb_images = rgb_images.unsqueeze(1)
    dist = torch.sum((rgb_images - palette_normalized) ** 2, dim=2)
    min_dist, class_maps = torch.min(dist, dim=1)
    ignore_mask = min_dist > threshold
    class_maps[ignore_mask] = 255
    return class_maps

def class_id_to_rgb(class_ids, palette):
    if not isinstance(palette, torch.Tensor):
        palette = torch.tensor(palette, dtype=torch.uint8)
    palette = palette.to(class_ids.device)
    if palette.dim() == 1:
        palette = palette.view(-1, 3)
    rgb_images = torch.zeros((class_ids.shape[0], 3, class_ids.shape[1], class_ids.shape[2]), dtype=torch.uint8,
                             device=class_ids.device)
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

@torch.no_grad()
def test(args):
    # 使用相对路径
    base_dir = "work_dirs/semflow/checkpoint-36000"
    
    # 创建可视化目录
    vis_dir = Path(base_dir) / "vis_test"
    vis_dir.mkdir(exist_ok=True, parents=True)

    # 加载测试数据加载器
    test_dataloader = pr_val_dataloader(args)

    # 加载模型
    args.pretrain_model = 'dataset/pretrain/stable-diffusion-v1-5'
    args.vae_path = 'dataset/pretrain/stable-diffusion-v1-5/vae'
    vae = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=None)

    # Accelerator
    accelerator = Accelerator()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # 设置设备和数据类型
    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # 使用accelerator准备模型
    vae, unet = accelerator.prepare(vae, unet)

    # 从指定路径加载UNet模型
    try:
        unwrapped_unet = accelerator.unwrap_model(unet)

        # 构建模型文件路径
        model_path = os.path.join(base_dir, "pytorch_model", "model_states.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # 加载.pt格式的模型权重
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # 检查文件内容类型
        if isinstance(checkpoint, dict):
            # 如果是字典，检查是否是完整检查点
            if 'module' in checkpoint:
                # 完整检查点，包含模型、优化器等
                unet_state_dict = checkpoint['module']
                logger.info("Loaded model weights from 'module' key in checkpoint")
            elif 'model_state_dict' in checkpoint:
                unet_state_dict = checkpoint['model_state_dict']
                logger.info("Loaded model weights from 'model_state_dict' key in checkpoint")
            elif 'state_dict' in checkpoint:
                unet_state_dict = checkpoint['state_dict']
                logger.info("Loaded model weights from 'state_dict' key in checkpoint")
            else:
                # 假设整个字典就是状态字典
                unet_state_dict = checkpoint
                logger.info("Using entire checkpoint as state dict")
        else:
            # 如果不是字典，可能是直接的模型状态字典
            logger.error(f"Unexpected checkpoint format: {type(checkpoint)}")
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
        
        # 移除可能的'module.'前缀（如果模型是在DataParallel中训练的）
        unet_state_dict = {k.replace('module.', ''): v for k, v in unet_state_dict.items()}
        
        # 加载权重
        missing_keys, unexpected_keys = unwrapped_unet.load_state_dict(unet_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading UNet: {len(missing_keys)} keys")
            # 只显示前几个缺失的键，避免日志过长
            if len(missing_keys) > 10:
                logger.warning(f"First 10 missing keys: {missing_keys[:10]}")
            else:
                logger.warning(f"Missing keys: {missing_keys}")
                
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading UNet: {len(unexpected_keys)} keys")
            # 只显示前几个意外的键，避免日志过长
            if len(unexpected_keys) > 10:
                logger.warning(f"First 10 unexpected keys: {unexpected_keys[:10]}")
            else:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        logger.info(f"Successfully loaded UNet weights from {model_path}")

    except Exception as e:
        logger.error(f"Could not load UNet weights: {e}")
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")

    # 在模型之后准备测试数据加载器以处理分片
    test_dataloader = accelerator.prepare(test_dataloader)

    unet.eval()

    # 初始化指标
    dataset = test_dataloader.dataset
    iou_metric = IoU(num_classes=dataset.num_classes, ignore_index=dataset.ignore_index)

    progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process, desc="Evaluating")

    # 获取空条件
    from module.data.prepare_text import sd_null_condition
    null_condition = sd_null_condition(args.pretrain_model).to(device, dtype=weight_dtype)
    prompt_embeds, unet_added_conditions = get_unet_added_conditions(args, null_condition)

    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    timesteps = torch.arange(1, 1000, 1000 // num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)

    palette = torch.tensor(build_palette(args.pa.k, args.pa.s), dtype=torch.uint8)

    for i, batch in enumerate(progress_bar):
        images = batch['image_semseg'].to(dtype=weight_dtype, device=device)
        bsz = images.shape[0]
        image_latents = vae.encode(images).latent_dist.mode() * vae.config.scaling_factor

        encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)

        pred_latents, _ = pipeline_rf_reverse(timesteps, unet, image_latents, encoder_hidden_states, prompt_embeds,
                                              guidance_scale, None)

        pred_rgb = vae.decode(pred_latents / vae.config.scaling_factor).sample

        # 将预测的3通道RGB图像转换回1通道类别ID图
        pred_classes = rgb_to_class_id(pred_rgb, palette, device)

        # 真实值是输入的'image_semseg'，也是一个RGB图像
        target_classes = rgb_to_class_id(images, palette, device)

        # 如果存在通道维度则压缩
        if target_classes.dim() == 4 and target_classes.shape[1] == 1:
            target_classes = target_classes.squeeze(1)

        pred_classes_np = pred_classes.cpu().numpy()
        target_np = target_classes.cpu().numpy()

        iou_metric.add(pred_classes_np, target_np)

        # 随机保存一些预测和真实分割图用于可视化检查
        if random.random() < 1:
            idx = random.randint(0, bsz - 1)

            # 将类别ID图转换为RGB用于可视化
            pred_seg_map_rgb = class_id_to_rgb(pred_classes[idx:idx + 1], palette)
            target_seg_map_rgb = class_id_to_rgb(target_classes[idx:idx + 1], palette)

            # 保存图像
            save_image(pred_seg_map_rgb, vis_dir / f"batch_{i}_idx_{idx}_pred.png")
            save_image(target_seg_map_rgb, vis_dir / f"batch_{i}_idx_{idx}_gt.png")

    miou = iou_metric.get_miou()
    iou = iou_metric.get_iou()

    logger.info("***** Test results *****")
    logger.info(f"mIoU: {miou:.4f}")
    for i, class_iou in enumerate(iou):
        logger.info(f"Class {i} IoU: {class_iou:.4f}")


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args, cli_config)
    test(args)