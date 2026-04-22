import sys
sys.path.append('.')

import logging
import math
import os

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin
 
from tqdm import tqdm
from omegaconf import OmegaConf
import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel,HunyuanDiT2DModel ,Transformer2DModel,SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from module.data.load_dataset import pr_val_dataloader, pr_train_dataloader
from module.pipe.val import valrf
from module.data.hook import resume_state, save_normal
from module.data.prepare_text import sd_null_condition
import module.pipe.triplet_loss
from module.data.prepare_text_hunyuan import get_hunyuan_text_embeddings # 确保导入了这个函数
from diffusers.models.embeddings import get_2d_rotary_pos_embed 
#from module.pipe.unet_new import UNetModel_newpreview
logger = get_logger(__name__)


'''
这里rgb_latents是从图像编码得到的，latents是从mask编码得到的
'''

@torch.no_grad()
def pre_rf(data, vae, device, weight_dtype, args):
    rgb_images = data['image'].to(dtype=weight_dtype, device=device)
    images = data['image_semseg'].to(dtype=weight_dtype, device=device)
    noise = torch.rand_like(images)-0.5
    images = images + args.pert.co*noise
    # 原代码：rgb_latents = vae.encode(rgb_images).latent_dist.sample()* vae.scaling_factor
    rgb_latents = vae.encode(rgb_images).latent_dist.sample()* vae.config.scaling_factor
    # 原代码：latents = vae.encode(images).latent_dist.mode()* vae.scaling_factor
    latents = vae.encode(images).latent_dist.mode()* vae.config.scaling_factor

    return latents, rgb_latents


def main(args):

    ttlsteps = 1000
    args.transformation.size = args.env.size

    print('init SD 1.5')
    #args.pretrain_model = '/home/ldp/LXW/SemFlow-xu/old/dataset/pretrain/stable-diffusion-v1-5'
    args.vae_path = '/home/ldp/LXW/SemFlow-xu/old/dataset/pretrain/stable-diffusion-v1-5/vae'
    args.pretrain_model = '/home/ldp/LXW/SemFlow-xu/delta-FM/TransSegFlow/dataset/pretrain_dit'
    #args.vae_path = '/home/ldp/LXW/SemFlow-xu/delta-FM/TransSegFlow/dataset/pretrain_dit/vae'
    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)
   
    logging_dir = Path(args.env.output_dir, args.env.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.env.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.env.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin if args.env.deepspeed else None
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.env.seed is not None:
        set_seed(args.env.seed)


    if accelerator.is_main_process:
        if args.env.output_dir is not None:
            os.makedirs(os.path.join(args.env.output_dir,'vis'), exist_ok=True)
            OmegaConf.save(args,os.path.join(args.env.output_dir,'config.yaml'))
    (
    encoder_hidden_states, 
    text_embedding_mask, 
    encoder_hidden_states_t5, 
    text_embedding_mask_t5
    ) = get_hunyuan_text_embeddings(
    prompt="找出透明物体,包括'Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window','Glass Door','Eyeglass' ,'Cup', 'Floor Glass', 'Glass Bow', 'Water Bottle', 'Storage Box",
    model_root_path=args.pretrain_model,
    device=accelerator.device,
    max_length_bert=77, 
    max_length_t5=256
    )
    # 确保嵌入数据在正确的设备上，并且不需要梯度
    encoder_hidden_states = encoder_hidden_states.detach()
    encoder_hidden_states_t5 = encoder_hidden_states_t5.detach()

    # 【关键】强制清理显存，把 T5 模型留下的垃圾彻底清空
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("文本嵌入计算完成，T5 模型显存已释放。")
    
    vae = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    unet =HunyuanDiT2DModel.from_pretrained(args.pretrain_model, subfolder="transformer")
    
#     unet = UNetModel_newpreview(
#     image_size=64,  # Stable Diffusion v1.5 使用的潜空间尺寸
#     in_channels=4,  # 潜空间的输入通道数
#     model_channels=128,
#     out_channels=4,
#     num_heads=4,
#     num_res_blocks=2, 
#     attention_resolutions=[16,8],
#     dropout=0,
#     channel_mult=[1, 2, 4, 8],
#     num_classes=None,
#     use_checkpoint=False,
#     num_head_channels=-1,
#     use_scale_shift_norm=True,
#     resblock_updown=False,
#     use_new_attention_order=False,
#     high_way=True  # 确保启用高速公路网络
# )
    # pretrained_unet_state_dict = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet").state_dict()
    # unet.load_part_state_dict(pretrained_unet_state_dict)
    # del pretrained_unet_state_dict # 释放内存
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
   
    vae.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

#    null_condition = sd_null_condition(args.pretrain_model)
   # null_condition = null_condition.to(accelerator.device, dtype=weight_dtype)
    
    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.env.scale_lr:
        args.optim.lr = (
            args.optim.lr * args.env.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    assert args.optim.name=='adamw'
    optimizer_class = torch.optim.AdamW  
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.optim.lr,
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )

    # # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    assert args.env.max_train_steps is not None

    lr_ratio = 1 if args.env.deepspeed else accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.warmup_steps * lr_ratio,
        num_training_steps=args.env.max_train_steps * lr_ratio,
    )

    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, val_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.env.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)

    ##
    if accelerator.is_main_process:
        accelerator.init_trackers("model")

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.env.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.env.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.env.max_train_steps}")
    global_step = 0
    first_epoch = 0

    first_epoch, resume_step, global_step = resume_state(accelerator,args,num_update_steps_per_epoch,unet)
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(global_step, args.env.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
###################################
    device = accelerator.device
    
    meta_list = [512, 512, 0, 0, 512, 512]
    image_meta_size = torch.tensor([meta_list] * 2, device=device, dtype=weight_dtype)
    style = torch.tensor([0] * 2, device=device, dtype=torch.long)
    rotary_emb_tuple = get_2d_rotary_pos_embed(
        embed_dim=88,
        crops_coords=((0, 0), (32, 32)), 
        grid_size=(32, 32),
        device=device,
        output_type="pt"
    )
    cos = rotary_emb_tuple[0].unsqueeze(0).unsqueeze(1).repeat(2, 1, 1, 1)
    sin = rotary_emb_tuple[1].unsqueeze(0).unsqueeze(1).repeat(2, 1, 1, 1)
    image_rotary_emb = (cos, sin)

    ###################
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            unet.train()
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.env.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                
                latents, z0 = pre_rf(batch,vae,device,weight_dtype,args)
                bsz = latents.shape[0]
                if args.cfg.continus:
                    t = torch.rand((bsz,),device=device,dtype=weight_dtype)
                    timesteps = t*ttlsteps
                else:
                    timesteps = torch.randint(0, ttlsteps,(bsz,), device=device,dtype=torch.long)
                    t = timesteps.to(weight_dtype)/ttlsteps

                t = t[:,None,None,None]
                perturb_latent = t*latents+(1.-t)*z0

                #prompt_embeds = null_condition.repeat(bsz,1,1)
                #model_pred = unet(perturb_latent, timesteps, prompt_embeds).sample
                model_pred = unet(
                hidden_states=perturb_latent,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states.repeat(bsz,1,1),
                text_embedding_mask=text_embedding_mask.repeat(bsz,1),
                encoder_hidden_states_t5=encoder_hidden_states_t5.repeat(bsz,1,1),
                text_embedding_mask_t5=text_embedding_mask_t5.repeat(bsz,1)  ,
                image_meta_size=image_meta_size, 
                style=style,
                image_rotary_emb=image_rotary_emb
                ).sample
               
                # 调用新模型############################
                #rgb_images = batch['image'].to(dtype=weight_dtype, device=device)
                #model_pred, cal_output = unet(perturb_latent, rgb_images, timesteps)
                
                ########################################
                target = latents - z0
                ##################################
                #对比损失compute_triplet_loss_efficiently
                #loss =module.pipe.triplet_loss.compute_triplet_loss_efficiently(model_pred.float(), target.float(),temperature=0.05)["loss"].mean()
                ##########################################
                if model_pred.shape[1] == 2 * target.shape[1]:
                    model_pred, _ = torch.split(model_pred, target.shape[1], dim=1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train.batch_size)).mean()
                train_loss += avg_loss.item() / args.env.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.env.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.env.checkpointing_steps == 0:
                    save_normal(accelerator,args,logger,global_step,unet)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # if args.env.val_iter > 0 and global_step % args.env.val_iter == 0:
            #     unet.eval()
                
            #     valrf(
            #         accelerator,
            #         args,
            #         vae,
            #         unet,
            #         val_dataloader,
            #         device,
            #         weight_dtype,
            #         null_condition,
            #         max_iter=None,
            #         gstep=global_step
            #         )

            if global_step >= args.env.max_train_steps:
                accelerator.wait_for_everyone()
                break

    accelerator.end_training()


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    assert os.path.isfile(cfg_path)
    # transsegflow 环境
    # args = OmegaConf.save(args,os.path.join(args.env.output_dir,'config.yaml')).OmegaConf.load(cfg_path)
    # yetbye 环境
    args = OmegaConf.load(cfg_path)
    # 确保输出目录存在
    os.makedirs(args.env.output_dir, exist_ok=True)
    config_save_path = os.path.join(args.env.output_dir, 'config.yaml')
    OmegaConf.save(args, config_save_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args,cli_config)
    main(args)