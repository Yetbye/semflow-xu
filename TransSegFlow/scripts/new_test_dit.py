import sys

sys.path.append('.')

import logging
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm import tqdm
import numpy as np
import os
import random
from torchvision.utils import save_image
from safetensors.torch import load_file
from module.data.load_dataset import pr_val_dataloader
from module.pipe.pipe_dit_o import pipeline_rf
from module.data.builder import build_palette
from einops import rearrange
from module.metrics.iou import IoU
#from module.metrics.Binary_iou import BinaryIoU
#from module.pipe.unet_new import UNetModel_newpreview  # 导入你的新模型
from module.pipe.DIT_META.dit_meta_o import DiT_models
from module.metrics.cls_evaluator import ClassificationEvaluator 
logger = get_logger(__name__)


def get_unet_added_conditions(args, null_condition):
    prompt_embeds = null_condition
    unet_added_conditions = None
    return prompt_embeds, unet_added_conditions


def rgb_to_class_id(rgb_images, palette, device, threshold=0.15):
    """
    Convert RGB images to class ID maps with a distance threshold.
    :param rgb_images: A tensor of shape (B, 3, H, W) with values in [-1, 1].
    :param palette: A tensor of shape (num_classes, 3) with values in [0, 255].
    :param device: The torch device.
    :param threshold: The maximum squared distance to consider a pixel as belonging to a class.
    :return: A tensor of shape (B, H, W) with class IDs.
    """
    # Convert palette to [-1, 1] range and reshape for broadcasting
    palette = (palette.to(device).float() / 127.5) - 1.0
    palette = palette.view(1, -1, 3, 1, 1)  # (1, num_classes, 3, 1, 1)

    # Reshape images for broadcasting: (B, 1, 3, H, W)
    rgb_images = rgb_images.unsqueeze(1)

    # Calculate L2 distance between each pixel and each palette color
    dist = torch.sum((rgb_images - palette) ** 2, dim=2)  # (B, num_classes, H, W)

    # Find the minimum distance and the corresponding class for each pixel
    min_dist, class_maps = torch.min(dist, dim=1)  # (B, H, W)

    # Pixels where the minimum distance is greater than the threshold are ignored
    # by setting their class ID to 255 (or another ignore_index)
    ignore_mask = min_dist > threshold
    class_maps[ignore_mask] = 255  # Assuming 255 is the ignore_index

    return class_maps


def class_id_to_rgb(class_ids, palette):
    """
    Convert class ID maps to RGB images.
    :param class_ids: A tensor of shape (B, H, W) with class IDs.
    :param palette: A tensor of shape (num_classes, 3) with values in [0, 255].
    :return: A tensor of shape (B, 3, H, W) with RGB values in [0, 1].
    """
    # Ensure palette is a tensor
    if not isinstance(palette, torch.Tensor):
        palette = torch.tensor(palette, dtype=torch.uint8)

    # Move palette to the same device as class_ids
    palette = palette.to(class_ids.device)

    # Reshape palette to (num_classes, 3)
    if palette.dim() == 1:
        palette = palette.view(-1, 3)

    # Create an empty tensor for the RGB images
    rgb_images = torch.zeros((class_ids.shape[0], 3, class_ids.shape[1], class_ids.shape[2]), dtype=torch.uint8,
                             device=class_ids.device)

    # Map class IDs to colors
    for class_id in range(palette.shape[0]):
        mask = class_ids == class_id
        rgb_images[:, 0, :, :][mask] = palette[class_id, 0]
        rgb_images[:, 1, :, :][mask] = palette[class_id, 1]
        rgb_images[:, 2, :, :][mask] = palette[class_id, 2]

    # Handle ignore_index (e.g., 255) by setting it to a specific color, e.g., black
    ignore_mask = class_ids == 255
    rgb_images[:, 0, :, :][ignore_mask] = 0
    rgb_images[:, 1, :, :][ignore_mask] = 0
    rgb_images[:, 2, :, :][ignore_mask] = 0

    return rgb_images.float() / 255.0


@torch.no_grad()
def test(args):
    # Create directory for saving visualizations
    vis_dir = Path(args.env.output_dir) / "vis_test"
    vis_dir.mkdir(exist_ok=True)
    cls_evaluator = ClassificationEvaluator(output_dir=vis_dir)
    # Load test dataloader
    test_dataloader = pr_val_dataloader(args)

    # Load models
    #args.pretrain_model = 'dataset/pretrain/stable-diffusion-v1-5'
    args.vae_path = 'dataset/pretrain_dit_hunyuan/vae'
    vae = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    #unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=None)
#     unet = UNetModel_newpreview(
#     image_size=64,
#     in_channels=4,
#     model_channels=128,
#     out_channels=4,
#     num_heads=4,
#     num_res_blocks=2,
#     attention_resolutions=[16, 8],
#     dropout=0,
#     channel_mult=[1, 2, 4, 8],
#     num_classes=None,
#     use_checkpoint=False,
#     num_head_channels=-1,
#     use_scale_shift_norm=True,
#     resblock_updown=False,
#     use_new_attention_order=False,
#     high_way=True
# )
    unet=DiT_models["DiT-XL/2"](input_size=64, num_classes=1000)
    #checkpoint = torch.load("dataset/pretrain_dit_meta/DiT-XL-2-512x512.pt", map_location="cpu")
    checkpoint = load_file("old_weight/checkpoint-80000-dit/model.safetensors", device="cpu")
    unet.load_state_dict(checkpoint,strict=False)
    # Accelerator
    accelerator = Accelerator()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set device and dtype
    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Prepare models with accelerator
    vae, unet = accelerator.prepare(vae, unet)

    # Load the UNet model from the specified checkpoint
    try:
       # unwrapped_unet = accelerator.unwrap_model(unet)

        # Construct the path to the model file
        model_path = os.path.join(args.env.output_dir, "model.safetensors")
        if not os.path.exists(model_path):
            # Fallback to pytorch_model.bin if safetensors is not found
            model_path = os.path.join(args.env.output_dir, "pytorch_model.bin")

        if os.path.exists(model_path):
            if model_path.endswith(".safetensors"):
                unet_state_dict = load_file(model_path, device="cpu")
            else:
                unet_state_dict = torch.load(model_path, map_location="cpu")

            #unwrapped_unet.load_state_dict(unet_state_dict)
            unet.load_state_dict(unet_state_dict)
            logger.info(f"Successfully loaded UNet weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found in {args.env.output_dir}")

    except Exception as e:
        logger.warning(
            f"Could not load UNet weights from '{args.env.output_dir}'. Evaluating with the base pre-trained UNet model. Error: {e}")

    # The test dataloader is prepared after the model to handle sharding
    test_dataloader = accelerator.prepare(test_dataloader)

    unet.eval()

    # Initialize metric
    dataset = test_dataloader.dataset
    iou_metric = IoU(num_classes=dataset.num_classes, ignore_index=dataset.ignore_index)

    progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process, desc="Evaluating")
    total_correct = 0
    total_pixels = 0
    ignore_index = dataset.ignore_index
    # Get null condition
    # from module.data.prepare_text import sd_null_condition
    # null_condition = sd_null_condition(args.pretrain_model).to(device, dtype=weight_dtype)
    # prompt_embeds, unet_added_conditions = get_unet_added_conditions(args, null_condition)

    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    
    # 使用demo.py中的timesteps处理方式
    timesteps = torch.arange(1, 1000, 1000 // num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)

    palette = torch.tensor(build_palette(args.pa.k, args.pa.s), dtype=torch.uint8)

    for i, batch in enumerate(progress_bar):
        images = batch['image'].to(dtype=weight_dtype, device=device)
        bsz = images.shape[0]
        image_latents = vae.encode(images).latent_dist.mode() * vae.config.scaling_factor

       # encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)

        # 使用demo.py中的分割流程
        y_labels = torch.full((bsz,), 1000, device=device, dtype=torch.long)
        #y_labels=batch['condition_image'].to(device=device,dtype=weight_dtype)
        pred_latents, _ = pipeline_rf(timesteps, unet, image_latents,guidance_scale, images,y_labels,None)
        pred_rgb = vae.decode(pred_latents / vae.config.scaling_factor).sample

        # Convert the predicted 3-channel RGB image back to a 1-channel class ID map.
        pred_classes = rgb_to_class_id(pred_rgb, palette, device)

        # Convert the ground truth RGB segmentation image back to a class ID map
        target_classes = rgb_to_class_id(batch['image_semseg'].to(device=device, dtype=weight_dtype), palette, device)

        pred_classes_np = pred_classes.cpu().numpy()
        target_np = target_classes.cpu().numpy()

        iou_metric.add(pred_classes_np, target_np)
        # [新增] 计算当个 batch 的有效像素和预测正确的像素
        valid_mask = target_np != ignore_index
        total_correct += np.sum((pred_classes_np == target_np) & valid_mask)
        total_pixels += np.sum(valid_mask)
        # Randomly save some predicted and ground truth segmentation maps for visual inspection
        if random.random() < 1.2:  # 20% probability to save
            idx = random.randint(0, bsz - 1)

            # Convert class ID maps to RGB for visualization
            #pred_seg_map_rgb = class_id_to_rgb(pred_classes[idx:idx + 1], palette)
            #target_seg_map_rgb = class_id_to_rgb(target_classes[idx:idx + 1], palette)

            # 使用demo.py中的颜色映射方式
            CLASS_COLORS = [
                (0, 0, 0), (120, 120, 70), (235, 235, 7),
                (6, 230, 230), (204, 255, 4), (120, 120, 120),
                (140, 140, 140), (255, 51, 7), (224, 5, 255),
                (204, 5, 255), (150, 5, 61), (4, 250, 7)
            ]
            # 构造保存时使用的调色板
            save_palette = torch.tensor(CLASS_COLORS, dtype=torch.uint8)
            # 重新生成可视化图：使用自定义颜色
            pred_vis = class_id_to_rgb(pred_classes[idx:idx + 1], save_palette)
            target_vis = class_id_to_rgb(target_classes[idx:idx + 1], save_palette)
            save_image(pred_vis, vis_dir / f"batch_{i}_idx_{idx}_pred.png")
            save_image(target_vis, vis_dir / f"batch_{i}_idx_{idx}_gt.png")

    miou = iou_metric.get_miou()
    iou = iou_metric.get_iou()
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    # 关闭文件句柄
    logger.info("***** Test results *****")
    logger.info(f"mIoU: {miou:.4f}")
    # [新增] 打印 Pixel Accuracy
    logger.info(f"Pixel Accuracy: {pixel_acc:.4f}")
    for i, class_iou in enumerate(iou):
        logger.info(f"Class {i} IoU: {class_iou:.4f}")


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args, cli_config)
    test(args)