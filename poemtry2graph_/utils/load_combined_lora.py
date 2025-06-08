#!/usr/bin/env python3
"""
结合UNet和Text Encoder LoRA的推理脚本
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_combined_lora_model():
    """加载结合了UNet和Text Encoder LoRA的模型"""
    
    BASE_MODEL_PATH = "./models/diffusions"
    UNET_LORA_PATH = "./fixed_lora_new"
    TEXT_ENCODER_LORA_PATH = "./text_encoder_10epoch"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"🚀 加载结合LoRA模型到设备: {DEVICE}")
    
    # 加载基础模型
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True
    )
    
    # 1. 加载UNet LoRA权重
    print("📦 加载UNet LoRA权重...")
    pipeline.load_lora_weights(UNET_LORA_PATH)
    
    # 2. 加载Text Encoder LoRA权重
    print("📝 加载Text Encoder LoRA权重...")
    try:
        # 使用PEFT加载text encoder的LoRA
        pipeline.text_encoder = PeftModel.from_pretrained(
            pipeline.text_encoder,
            TEXT_ENCODER_LORA_PATH,
            torch_dtype=torch.float32
        )
        print("✅ Text Encoder LoRA加载成功")
    except Exception as e:
        print(f"⚠️  Text Encoder LoRA加载失败: {e}")
        print("🔄 尝试使用diffusers方式加载...")
        # 备选方案：使用diffusers的load_lora_weights方法
        try:
            pipeline.load_lora_weights(TEXT_ENCODER_LORA_PATH, adapter_name="text_encoder")
            pipeline.set_adapters(["default", "text_encoder"], adapter_weights=[1.0, 1.0])
            print("✅ Text Encoder LoRA备选方案加载成功")
        except Exception as e2:
            print(f"❌ Text Encoder LoRA备选方案也失败: {e2}")
    
    pipeline = pipeline.to(DEVICE)
    
    print("✅ 结合LoRA模型加载完成")
    return pipeline

def generate_image(pipeline, prompt, output_path="combined_lora_output.png"):
    """使用结合LoRA模型生成图像"""
    
    DEVICE = pipeline.device
    
    print(f"🎨 生成图像: {prompt}")
    
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

def test_different_prompts(pipeline):
    """测试不同的提示词，对比效果"""
    
    test_prompts = [
        "日往菲薇，月来扶疏。",
        "雷雨窈冥而未半，皦日笼光於绮寮。",
        "蕙风如薰，甘露如醴。",
        "槟榔无柯，椰叶无阴。",
        "绿叶翠茎，冒霜停雪。"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        output_path = f"combined_lora_test_{i}_10_epoch.png"
        generate_image(pipeline, prompt, output_path)
        print(f"完成测试 {i}/4")

if __name__ == "__main__":
    # 加载结合模型
    combined_pipeline = load_combined_lora_model()
    
    # 单个测试
    test_prompt = "a river and a mountain"
    generate_image(combined_pipeline, test_prompt, "combined_lora_single_test.png")
    
    # 多个测试
    #print("\n🔍 开始多样化测试...")
    #test_different_prompts(combined_pipeline)
    
    print("\n🎉 结合LoRA模型测试完成！")
    print("📊 生成的图像文件:")
    #print("  - combined_lora_single_test.png")
    #print("  - combined_lora_test_1.png")
    #print("  - combined_lora_test_2.png") 
    #print("  - combined_lora_test_3.png")
    #print("  - combined_lora_test_4.png") 