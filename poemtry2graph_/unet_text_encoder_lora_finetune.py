#!/usr/bin/env python3
"""
Stable Diffusion v1.5 UNet + Text Encoder LoRA联合微调脚本 - A800超算优化版本
针对NVIDIA A800 80GB显存优化，同时训练UNet和Text Encoder的LoRA适配器
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model_state_dict
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import logging
import time
import gc

# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="UNet + Text Encoder LoRA联合微调")
    
    # 模型和数据路径
    parser.add_argument("--model_id", type=str, default="./models/diffusions", 
                       help="本地Stable Diffusion模型路径")
    parser.add_argument("--csv_path", type=str, default="./refined_image/image_prompts.csv",
                       help="CSV数据文件路径")
    parser.add_argument("--image_dir", type=str, default="./refined_image//",
                       help="图片文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./lora_poem_style_joint_new",
                       help="LoRA权重输出目录")
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="UNet LoRA学习率")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-5,
                       help="Text Encoder LoRA学习率")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="梯度累积步数")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # 其他参数
    parser.add_argument("--resolution", type=int, default=512,
                       help="图像分辨率")
    parser.add_argument("--max_text_length", type=int, default=77,
                       help="文本最大长度")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="启用自动混合精度")
    parser.add_argument("--compile_model", action="store_true", default=True,
                       help="PyTorch 2.0编译优化")
    
    return parser.parse_args()

# ==================== A800优化数据集类 ====================
class PoemImageDataset(torch.utils.data.Dataset):
    """针对A800优化的诗歌-图像数据集类"""
    
    def __init__(self, csv_path, image_dir, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # 读取CSV数据
        logging.info("正在加载CSV数据...")
        self.data = pd.read_csv(csv_path, sep='\t')
        self.image_dir = Path(image_dir)
        
        # 预先建立所有图片路径映射
        logging.info("正在建立图片路径映射...")
        self.id_to_path = {}
        supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        
        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() in supported_formats:
                self.id_to_path[img_path.stem] = img_path
        
        logging.info(f"找到 {len(self.id_to_path)} 个图片文件")
        
        # 过滤掉没有对应图片的数据
        valid_indices = []
        for idx, row in self.data.iterrows():
            if row['image_id'] in self.id_to_path:
                valid_indices.append(idx)
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        logging.info(f"过滤后有效数据: {len(self.data)} 个样本")
        
        # A800优化的图像预处理 - 添加数据增强
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.1),  # 轻微数据增强
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        logging.info(f"数据集初始化完成，共 {len(self.data)} 个有效样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # 获取数据行
            row = self.data.iloc[idx]
            image_id = row['image_id']
            poem_text = str(row['poem'])  # 确保是字符串
            
            # 加载并处理图像
            img_path = self.id_to_path[image_id]
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.image_transforms(image)
            
            # 文本编码
            text_inputs = self.tokenizer(
                poem_text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0)
            }
            
        except Exception as e:
            logging.warning(f"处理样本 {idx} 时出错: {e}")
            # 返回第一个样本作为fallback
            return self.__getitem__(0)

# ==================== A800优化工具函数 ====================
def unwrap_model(model):
    """解包编译后的模型"""
    # 如果模型被torch.compile包装，需要访问原始模型
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model

def setup_unet_lora(unet, lora_config):
    """为UNet设置LoRA适配器"""
    logging.info("为UNet配置LoRA适配器...")
    
    # 添加LoRA适配器
    unet.add_adapter(lora_config)
    
    # 设置训练模式
    unet.train()
    
    # 冻结非LoRA参数
    unet.requires_grad_(False)
    
    # 只启用LoRA参数的梯度
    lora_params = 0
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_params += param.numel()
    
    total_params = sum(p.numel() for p in unet.parameters())
    logging.info(f"UNet总参数: {total_params:,}")
    logging.info(f"UNet LoRA参数: {lora_params:,}")
    logging.info(f"UNet可训练参数比例: {100 * lora_params / total_params:.3f}%")
    
    return unet

def setup_text_encoder_lora(text_encoder, lora_config):
    """为Text Encoder设置LoRA适配器"""
    logging.info("为Text Encoder配置LoRA适配器...")
    
    # 添加LoRA适配器
    text_encoder.add_adapter(lora_config)
    
    # 设置训练模式
    text_encoder.train()
    
    # 冻结非LoRA参数
    text_encoder.requires_grad_(False)
    
    # 只启用LoRA参数的梯度
    lora_params = 0
    for name, param in text_encoder.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_params += param.numel()
    
    total_params = sum(p.numel() for p in text_encoder.parameters())
    logging.info(f"Text Encoder总参数: {total_params:,}")
    logging.info(f"Text Encoder LoRA参数: {lora_params:,}")
    logging.info(f"Text Encoder可训练参数比例: {100 * lora_params / total_params:.3f}%")
    
    return text_encoder

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU内存: {allocated:.2f}GB已分配, {reserved:.2f}GB已保留"
    return "GPU不可用"

# ==================== A800优化训练函数 ====================
def main():
    args = parse_args()
    
    # 设备和数据类型配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 对于A800，统一使用float32以避免类型不匹配，显存充足
    DTYPE = torch.float32
    COMPUTE_DTYPE = torch.float16 if args.mixed_precision else torch.float32
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.output_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 设置CUDA优化
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logging.info("=" * 60)
    logging.info("开始A800优化UNet+Text Encoder联合LoRA微调训练...")
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"数据类型: {DTYPE}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"UNet学习率: {args.learning_rate}")
    logging.info(f"Text Encoder学习率: {args.text_encoder_lr}")
    logging.info(f"LoRA秩: {args.lora_rank}")
    logging.info("=" * 60)
    
    start_time = time.time()
    
    # 1. 加载预训练模型组件
    logging.info("加载预训练模型组件...")
    
    # 加载tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    
    # 加载UNet - 直接指定数据类型
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, 
        subfolder="unet",
        torch_dtype=DTYPE
    )
    
    # 加载Text Encoder - 直接指定数据类型
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=DTYPE
    )
    
    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # 加载VAE
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=DTYPE,
        safety_checker=None,  # 禁用安全检查器以节省显存
        requires_safety_checker=False
    )
    
    vae = pipeline.vae.eval()
    
    # 清理pipeline以节省显存
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 配置LoRA
    # UNet LoRA配置
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "conv1", "conv2", "conv_shortcut",
            "proj_in", "proj_out"
        ],
        lora_dropout=args.lora_dropout,
    )
    
    # Text Encoder LoRA配置
    text_encoder_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2"
        ],
        lora_dropout=args.lora_dropout,
    )
    
    # 设置模型LoRA
    unet = setup_unet_lora(unet, unet_lora_config)
    text_encoder = setup_text_encoder_lora(text_encoder, text_encoder_lora_config)
    
    # 冻结VAE
    vae.requires_grad_(False)
    
    # 3. 移动模型到设备
    logging.info("移动模型到GPU...")
    unet = unet.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)
    vae = vae.to(DEVICE)
    
    # 4. 编译模型优化 (PyTorch 2.0)
    if args.compile_model and hasattr(torch, 'compile'):
        logging.info("编译模型以提升性能...")
        unet = torch.compile(unet)
        text_encoder = torch.compile(text_encoder)
    
    logging.info(get_gpu_memory_info())
    
    # 5. 准备数据集和数据加载器
    logging.info("准备数据集...")
    dataset = PoemImageDataset(args.csv_path, args.image_dir, tokenizer, args.resolution)
    
    # A800优化的DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True  # 确保批次大小一致
    )
    
    logging.info(f"数据加载器配置: {len(dataloader)} 个批次")
    
    # 6. 设置优化器和调度器
    # 分别收集UNet和Text Encoder的LoRA参数
    unet_lora_params = [p for n, p in unet.named_parameters() if p.requires_grad]
    text_encoder_lora_params = [p for n, p in text_encoder.named_parameters() if p.requires_grad]
    
    # 使用不同学习率的参数组
    optimizer = torch.optim.AdamW([
        {'params': unet_lora_params, 'lr': args.learning_rate, 'name': 'unet'},
        {'params': text_encoder_lora_params, 'lr': args.text_encoder_lr, 'name': 'text_encoder'}
    ], betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
    
    logging.info(f"UNet LoRA参数数量: {len(unet_lora_params)}")
    logging.info(f"Text Encoder LoRA参数数量: {len(text_encoder_lora_params)}")
    
    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 7. 设置自动混合精度
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
    
    # 8. 训练循环
    logging.info("开始训练循环...")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移动到设备，确保数据类型一致
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            
            # 使用自动混合精度
            with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=COMPUTE_DTYPE):
                with torch.no_grad():
                    # 编码图像到潜空间
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # 编码文本 - 现在Text Encoder参与训练
                encoder_hidden_states = text_encoder(
                    input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
                
                # 采样噪声 - 确保数据类型一致
                noise = torch.randn_like(latents, dtype=latents.dtype)
                bsz = latents.shape[0]
                
                # 随机采样时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                    dtype=torch.long
                )
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"未知预测类型: {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred, target, reduction="mean")
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积和优化
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.6f}",
                "unet_lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "te_lr": f"{optimizer.param_groups[1]['lr']:.2e}",
                "step": global_step
            })
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Epoch结束处理
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # 更新学习率
        scheduler.step()
        
        logging.info(
            f"Epoch {epoch+1}/{args.num_epochs} 完成 | "
            f"平均损失: {avg_loss:.6f} | "
            f"时间: {epoch_time:.2f}s | "
            f"UNet学习率: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Text Encoder学习率: {optimizer.param_groups[1]['lr']:.2e}"
        )
        logging.info(get_gpu_memory_info())
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lora_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_lora_dir, exist_ok=True)
            
            # 解包编译后的模型
            unwrapped_unet = unwrap_model(unet)
            unwrapped_text_encoder = unwrap_model(text_encoder)
            
            # 获取LoRA权重
            unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
            text_encoder_lora_state_dict = get_peft_model_state_dict(unwrapped_text_encoder)
            
            # 保存LoRA权重
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                best_lora_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_state_dict,
            )
            
            logging.info(f"保存最佳LoRA权重到: {best_lora_dir}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 解包编译后的模型
            unwrapped_unet = unwrap_model(unet)
            unwrapped_text_encoder = unwrap_model(text_encoder)
            
            # 获取LoRA权重
            unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
            text_encoder_lora_state_dict = get_peft_model_state_dict(unwrapped_text_encoder)
            
            # 保存LoRA权重
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                checkpoint_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_state_dict,
            )
            
            # 保存训练状态
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            logging.info(f"检查点已保存到: {checkpoint_dir}")
    
    # 9. 保存最终LoRA权重
    logging.info("保存最终LoRA权重...")
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 解包编译后的模型
    unwrapped_unet = unwrap_model(unet)
    unwrapped_text_encoder = unwrap_model(text_encoder)
    
    # 获取LoRA权重
    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
    text_encoder_lora_state_dict = get_peft_model_state_dict(unwrapped_text_encoder)
    
    # 保存LoRA权重
    StableDiffusionLoraLoaderMixin.save_lora_weights(
        final_output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
    )
    
    total_time = time.time() - start_time
    logging.info("=" * 60)
    logging.info(f"训练完成！总用时: {total_time/3600:.2f} 小时")
    logging.info(f"最佳损失: {best_loss:.6f}")
    logging.info(f"LoRA权重已保存到: {final_output_dir}")
    logging.info(f"最佳LoRA权重保存在: {os.path.join(args.output_dir, 'best_model')}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()