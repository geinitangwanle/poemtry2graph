#!/usr/bin/env python3
"""
Stable Diffusion v1.5 LoRA微调脚本 - A800超算优化版本
针对NVIDIA A800 80GB显存优化，解决数据类型不匹配问题
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer
from peft import LoraConfig
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import logging
import time
import gc

# ==================== A800超算优化参数配置 ====================
# 模型和数据路径
MODEL_ID = "./models/diffusions"              # 本地Stable Diffusion模型路径
CSV_PATH = "./lora_data/POEM_IMAGE.csv"       # CSV数据文件路径
IMAGE_DIR = "./lora_data/processed_image/"    # 图片文件夹路径
OUTPUT_DIR = "./lora_poem_style"              # LoRA权重输出目录

# A800优化训练参数 (充分利用80GB显存)
LEARNING_RATE = 1e-4                          # 学习率
NUM_EPOCHS = 50                               # 训练轮数
BATCH_SIZE = 8                                # A800可以支持更大批次
GRADIENT_ACCUMULATION_STEPS = 2               # 减少梯度累积，因为batch size增大了
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 有效批次大小：16

# LoRA参数
LORA_RANK = 16                               # 增大LoRA秩以提高表达能力
LORA_ALPHA = 32                              # 相应增大alpha
LORA_DROPOUT = 0.1                           # LoRA dropout

# 其他优化参数
RESOLUTION = 512                              # 图像分辨率
MAX_TEXT_LENGTH = 77                         # 文本最大长度
MIXED_PRECISION = True                       # 启用自动混合精度
COMPILE_MODEL = True                         # PyTorch 2.0编译优化
DATALOADER_NUM_WORKERS = 4                   # 多进程数据加载
PIN_MEMORY = True                            # 启用pin memory
PREFETCH_FACTOR = 4                          # 数据预取倍数

# 设备和数据类型配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 对于A800，统一使用float32以避免类型不匹配，显存充足
DTYPE = torch.float32
COMPUTE_DTYPE = torch.float16 if MIXED_PRECISION else torch.float32

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置CUDA优化
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==================== A800优化数据集类 ====================
class PoemImageDataset(torch.utils.data.Dataset):
    """针对A800优化的诗歌-图像数据集类"""
    
    def __init__(self, csv_path, image_dir, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # 读取CSV数据
        logger.info("正在加载CSV数据...")
        self.data = pd.read_csv(csv_path, sep='\t')
        self.image_dir = Path(image_dir)
        
        # 预先建立所有图片路径映射
        logger.info("正在建立图片路径映射...")
        self.id_to_path = {}
        supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        
        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() in supported_formats:
                self.id_to_path[img_path.stem] = img_path
        
        logger.info(f"找到 {len(self.id_to_path)} 个图片文件")
        
        # 过滤掉没有对应图片的数据
        valid_indices = []
        for idx, row in self.data.iterrows():
            if row['image_id'] in self.id_to_path:
                valid_indices.append(idx)
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"过滤后有效数据: {len(self.data)} 个样本")
        
        # A800优化的图像预处理 - 添加数据增强
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.1),  # 轻微数据增强
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        logger.info(f"数据集初始化完成，共 {len(self.data)} 个有效样本")
    
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
                max_length=MAX_TEXT_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0)
            }
            
        except Exception as e:
            logger.warning(f"处理样本 {idx} 时出错: {e}")
            # 返回第一个样本作为fallback
            return self.__getitem__(0)

# ==================== A800优化工具函数 ====================
def setup_model_for_training(unet, lora_config):
    """为A800优化模型设置"""
    logger.info("配置LoRA适配器...")
    
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
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"LoRA参数: {lora_params:,}")
    logger.info(f"可训练参数比例: {100 * lora_params / total_params:.3f}%")
    
    return unet

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU内存: {allocated:.2f}GB已分配, {reserved:.2f}GB已保留"
    return "GPU不可用"

# ==================== A800优化训练函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("开始A800优化LoRA微调训练...")
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"数据类型: {DTYPE}")
    logger.info(f"批次大小: {BATCH_SIZE}")
    logger.info(f"有效批次大小: {EFFECTIVE_BATCH_SIZE}")
    logger.info(f"LoRA秩: {LORA_RANK}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 1. 加载预训练模型组件
    logger.info("加载预训练模型组件...")
    
    # 加载tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    
    # 加载UNet - 直接指定数据类型
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, 
        subfolder="unet",
        torch_dtype=DTYPE
    )
    
    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    # 加载VAE和Text Encoder
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,  # 禁用安全检查器以节省显存
        requires_safety_checker=False
    )
    
    vae = pipeline.vae.eval()
    text_encoder = pipeline.text_encoder.eval()
    
    # 清理pipeline以节省显存
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 配置LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "conv1", "conv2", "conv_shortcut",
            "proj_in", "proj_out"
        ],
        lora_dropout=LORA_DROPOUT,
    )
    
    # 设置模型
    unet = setup_model_for_training(unet, lora_config)
    
    # 冻结其他模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 3. 移动模型到设备
    logger.info("移动模型到GPU...")
    unet = unet.to(DEVICE)
    vae = vae.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)
    
    # 4. 编译模型优化 (PyTorch 2.0)
    if COMPILE_MODEL and hasattr(torch, 'compile'):
        logger.info("编译UNet模型以提升性能...")
        unet = torch.compile(unet)
    
    logger.info(get_gpu_memory_info())
    
    # 5. 准备数据集和数据加载器
    logger.info("准备数据集...")
    dataset = PoemImageDataset(CSV_PATH, IMAGE_DIR, tokenizer, RESOLUTION)
    
    # A800优化的DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
        drop_last=True  # 确保批次大小一致
    )
    
    logger.info(f"数据加载器配置: {len(dataloader)} 个批次")
    
    # 6. 设置优化器和调度器
    # 只优化LoRA参数
    lora_params = [p for n, p in unet.named_parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    
    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # 7. 设置自动混合精度
    scaler = torch.amp.GradScaler('cuda') if MIXED_PRECISION else None
    
    # 8. 训练循环
    logger.info("开始训练循环...")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移动到设备，确保数据类型一致
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            
            # 使用自动混合精度
            with torch.amp.autocast('cuda', enabled=MIXED_PRECISION, dtype=COMPUTE_DTYPE):
                with torch.no_grad():
                    # 编码图像到潜空间
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # 编码文本
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
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积和优化
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
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
        
        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} 完成 | "
            f"平均损失: {avg_loss:.6f} | "
            f"时间: {epoch_time:.2f}s | "
            f"学习率: {optimizer.param_groups[0]['lr']:.2e}"
        )
        logger.info(get_gpu_memory_info())
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_checkpoint_dir, exist_ok=True)
            unet.save_pretrained(best_checkpoint_dir)
            logger.info(f"保存最佳模型到: {best_checkpoint_dir}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            unet.save_pretrained(checkpoint_dir)
            
            # 保存训练状态
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            logger.info(f"检查点已保存到: {checkpoint_dir}")
    
    # 9. 保存最终模型
    logger.info("保存最终LoRA权重...")
    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    unet.save_pretrained(final_output_dir)
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"训练完成！总用时: {total_time/3600:.2f} 小时")
    logger.info(f"最佳损失: {best_loss:.6f}")
    logger.info(f"LoRA权重已保存到: {final_output_dir}")
    logger.info(f"最佳模型保存在: {os.path.join(OUTPUT_DIR, 'best_model')}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 