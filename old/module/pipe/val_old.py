import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from typing import Optional
from tqdm import tqdm
from .pipe import pipeline_rf_reverse
import numpy as np
import cv2
import numpy as np
from ..data.builder import build_palette
from einops import rearrange
from einops import rearrange
import os
import os.path as osp
import random


def get_unet_added_conditions(args,null_condition):
    prompt_embeds = null_condition
    unet_added_conditions = None
    return prompt_embeds, unet_added_conditions


def l2i(latents,vae,weight_dtype,file_names=None):
    ##
    # 原代码：latents = latents * (1. / vae.scaling_factor)
    latents = latents * (1. / vae.config.scaling_factor)
    masks_logits_tmp = vae.decode(latents.to(weight_dtype)).sample
    imgs = []
    names = []
    for i in range(latents.shape[0]):
        _tmp = masks_logits_tmp[i]*127.5+127.5
        _tmp = torch.clamp(_tmp,0,255)
        img = _tmp.detach().float().cpu().permute(1,2,0).numpy().astype(np.uint8)
        img = img[:,:,::-1]
        imgs.append(img)
        if file_names:
            name = file_names[i].split('/')[-1]
            names.append(name)
    return imgs, names


@torch.no_grad()
def valrf(
    accelerator: Accelerator,
    args,
    vae,
    unet,
    dataloader,
    device,
    weight_dtype,
    null_condition,
    num_inference_steps: int = None,
    max_iter: Optional[int] = None,
    gstep=0,
):

    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    palette = build_palette(args.pa.k,args.pa.s)
    meta_data = dataloader.dataset.meta_data    
    assert args.mode=='semantic'
    map_tbl = meta_data['category_names']
    cls_num = len(map_tbl)
    
    table = torch.tensor(palette[:cls_num*3])
    table = rearrange(table,'(q c) -> c q',c=3)
    table = table.to(device=device,dtype=weight_dtype)[None,:,:,None,None]  # 3,cls
    # 使用更精确的转换公式，确保与图像的[-1, 1]范围一致
    table = (table / 255.0) * 2.0 - 1.0

    prompt_embeds, unet_added_conditions = get_unet_added_conditions(args,null_condition)
    timesteps = torch.arange(1,1000,1000//num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps),-1).flip([0,1]).squeeze(1)

    # 添加保存控制参数
    save_probability = getattr(args, 'save_probability', 0.02)  # 保存概率，默认10%
    max_images_per_step = getattr(args, 'max_images_per_step', 5)  # 每个验证步骤最多保存图片数
    max_batches = getattr(args, 'max_val_batches', 1000)  # 最多验证批次数量

    saved_count = 0
    
    for batch_idx, data in enumerate(dataloader):
        # 限制验证批次数量
        if batch_idx >= max_batches:
            break
            
        file_names = [x["image_file"] for x in data['meta']]
        rgb_images = data['image'].to(device=device, dtype=weight_dtype)

        # 原代码：rgb_latents = vae.encode(rgb_images).latent_dist.mode()*vae.scaling_factor
        rgb_latents = vae.encode(rgb_images).latent_dist.mode()*vae.config.scaling_factor
        bsz = rgb_latents.shape[0]
        encoder_hidden_states = prompt_embeds.repeat(bsz,1,1)  

        if unet_added_conditions is not None:
            _unet_added_conditions = {"time_ids":unet_added_conditions["time_ids"].repeat(bsz,1),
                                      "text_embeds":unet_added_conditions["text_embeds"].repeat(bsz,1)}
        else:
            _unet_added_conditions = None

        if accelerator.is_main_process:
            image_semseg = data['image_semseg'].to(device=device, dtype=weight_dtype)
            # 去除有限扰动
            # noise = torch.rand_like(image_semseg)-0.5
            # image_semseg += args.pert.co * noise
            
            # 原代码：image_latents = vae.encode(image_semseg).latent_dist.mode()*vae.scaling_factor
            image_latents = vae.encode(image_semseg).latent_dist.mode()*vae.config.scaling_factor
            rlatents,_ = pipeline_rf_reverse(timesteps,unet,image_latents,encoder_hidden_states,prompt_embeds,guidance_scale,_unet_added_conditions)
            imgs,names = l2i(rlatents,vae,weight_dtype,file_names)
            
            # 优化保存策略：概率性保存 + 数量限制
            for i in range(len(imgs)):
                # 检查是否达到最大保存数量
                if saved_count >= max_images_per_step:
                    break
                    
                # 概率性保存
                if random.random() < save_probability:
                    fold=osp.join(args.env.output_dir,'vis')
                    os.makedirs(fold, exist_ok=True)
                    cv2.imwrite(f'{fold}/{gstep}_{names[i]}',imgs[i])
                    saved_count += 1
                    
                    # 达到最大数量后停止保存
                    if saved_count >= max_images_per_step:
                        break