#!/usr/bin/env python3
"""
简洁的LoRA模型加载脚本
"""

import torch
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

def load_lora_model():
    """加载LoRA微调模型"""
    
    BASE_MODEL_PATH = "./models/diffusions"
    LORA_PATH = "./fixed_lora"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"🚀 加载LoRA模型到设备: {DEVICE}")
    
    # 加载基础模型
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True
    )
    
    # 加载LoRA权重
    pipeline.load_lora_weights(LORA_PATH)
    pipeline = pipeline.to(DEVICE)
    
    print("✅ LoRA模型加载完成")
    return pipeline

def generate_image(pipeline, prompt, output_path="lora_output.png"):
    """使用LoRA模型生成图像"""
    
    DEVICE = pipeline.device
    
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator(device=DEVICE).manual_seed(42)
        ).images[0]
    
    image.save(output_path)
    print(f"✅ 图像已保存: {output_path}")
    return image

if __name__ == "__main__":
    # 加载模型
    lora_pipeline = load_lora_model()
    
    # 生成测试图像
    test_prompt = "a river and a mountain"
    generate_image(lora_pipeline, test_prompt, "lora_test1.png")
    
    print("🎉 LoRA模型测试完成！") 