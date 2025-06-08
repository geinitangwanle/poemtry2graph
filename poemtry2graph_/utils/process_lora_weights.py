#!/usr/bin/env python3
"""
从包含LoRA结构的完整UNet模型中提取纯LoRA权重
"""

import torch
from safetensors.torch import load_file, save_file
import os
import json
from collections import OrderedDict

def extract_lora_weights():
    """提取LoRA权重并保存为标准格式"""
    
    print("="*70)
    print("提取LoRA权重")
    print("="*70)
    
    FINETUNED_UNET_PATH = "./finetune_unet_new"
    OUTPUT_DIR = "./fixed_lora_new"
    
    try:
        # 1. 加载包含LoRA的完整模型
        print("1. 加载包含LoRA结构的模型...")
        safetensors_path = os.path.join(FINETUNED_UNET_PATH, "diffusion_pytorch_model.safetensors")
        full_state_dict = load_file(safetensors_path, device="cpu")
        print(f"✓ 加载了 {len(full_state_dict)} 个权重")
        
        # 2. 分析权重结构
        print("\n2. 分析权重结构...")
        lora_a_weights = {}
        lora_b_weights = {}
        base_weights = {}
        
        for key, value in full_state_dict.items():
            if "lora_A" in key:
                # 移除.default后缀，保留核心路径
                clean_key = key.replace(".default", "")
                lora_a_weights[clean_key] = value
            elif "lora_B" in key:
                clean_key = key.replace(".default", "")
                lora_b_weights[clean_key] = value
            elif "base_layer" in key:
                # 保存基础层权重
                clean_key = key.replace(".base_layer", "")
                base_weights[clean_key] = value
        
        print(f"  LoRA A权重: {len(lora_a_weights)}")
        print(f"  LoRA B权重: {len(lora_b_weights)}")
        print(f"  基础层权重: {len(base_weights)}")
        
        # 3. 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 4. 保存纯LoRA权重
        print("\n3. 保存纯LoRA权重...")
        lora_weights = {}
        lora_weights.update(lora_a_weights)
        lora_weights.update(lora_b_weights)
        
        if lora_weights:
            lora_path = os.path.join(OUTPUT_DIR, "pytorch_lora_weights.safetensors")
            save_file(lora_weights, lora_path)
            print(f"✓ LoRA权重已保存: {lora_path}")
            print(f"  包含 {len(lora_weights)} 个LoRA参数")
        
        # 5. 创建LoRA配置文件
        print("\n4. 创建LoRA配置文件...")
        lora_config = {
            "base_model_name_or_path": "./models/diffusions",
            "inference_mode": False,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "r": 16,
            "target_modules": [
                "to_k", "to_q", "to_v", "to_out.0",
                "conv1", "conv2", "conv_shortcut", 
                "proj_in", "proj_out"
            ],
            "task_type": "DIFFUSION"
        }
        
        config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(lora_config, f, indent=2)
        print(f"✓ LoRA配置已保存: {config_path}")
        
        # 6. 保存基础权重（如果需要）
        if base_weights:
            base_path = os.path.join(OUTPUT_DIR, "base_model_weights.safetensors")
            save_file(base_weights, base_path)
            print(f"✓ 基础权重已保存: {base_path}")
            print(f"  包含 {len(base_weights)} 个基础参数")
        
        return True, OUTPUT_DIR
        
    except Exception as e:
        print(f"✗ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_extracted_lora():
    """测试提取的LoRA权重"""
    
    print("\n" + "="*70)
    print("测试提取的LoRA权重")
    print("="*70)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        BASE_MODEL_PATH = "./models/diffusions"
        LORA_PATH = "./fixed_lora_new"
        DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # 1. 加载基础模型
        print("1. 加载基础模型...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        print("✓ 基础模型加载成功")
        
        # 2. 加载LoRA权重
        print("\n2. 加载提取的LoRA权重...")
        pipeline.load_lora_weights(LORA_PATH)
        print("✓ LoRA权重加载成功")
        
        # 3. 移动到设备
        pipeline = pipeline.to(DEVICE)
        
        # 4. 测试生成
        print("\n3. 测试图像生成...")
        test_prompt = "举头望明月，低头思故乡，古典诗意山水画"
        
        with torch.no_grad():
            image = pipeline(
                prompt=test_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=torch.Generator(device=DEVICE).manual_seed(42)
            ).images[0]
        
        # 5. 保存结果
        os.makedirs("extracted_lora_test", exist_ok=True)
        image.save("extracted_lora_test/lora_test_result.png")
        print("✓ 测试图像已生成并保存")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """比较原始模型和LoRA模型的效果"""
    
    print("\n" + "="*70)
    print("效果对比测试")
    print("="*70)
    
    BASE_MODEL_PATH = "./models/diffusions" 
    LORA_PATH = "./fixed_lora_new"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    test_prompt = "举头望明月，低头思故乡，古典诗意山水画"
    
    try:
        from diffusers import StableDiffusionPipeline
        import numpy as np
        from PIL import Image
        
        # 1. 原始模型生成
        print("1. 原始模型生成...")
        original_pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to(DEVICE)
        
        with torch.no_grad():
            original_image = original_pipeline(
                prompt=test_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=torch.Generator(device=DEVICE).manual_seed(42)
            ).images[0]
        
        os.makedirs("lora_final_comparison", exist_ok=True)
        original_image.save("lora_final_comparison/original.png")
        print("✓ 原始图像已保存")
        
        # 2. LoRA模型生成
        print("\n2. LoRA模型生成...")
        lora_pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        
        lora_pipeline.load_lora_weights(LORA_PATH)
        lora_pipeline = lora_pipeline.to(DEVICE)
        
        with torch.no_grad():
            lora_image = lora_pipeline(
                prompt=test_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=torch.Generator(device=DEVICE).manual_seed(42)
            ).images[0]
        
        lora_image.save("lora_final_comparison/lora.png")
        print("✓ LoRA图像已保存")
        
        # 3. 计算差异
        print("\n3. 计算图像差异...")
        orig_array = np.array(original_image)
        lora_array = np.array(lora_image)
        
        pixel_diff = np.mean(np.abs(orig_array.astype(float) - lora_array.astype(float)))
        print(f"平均像素差异: {pixel_diff:.2f}")
        
        if pixel_diff > 5:
            print("🎉 LoRA微调有效果！生成的图像有明显差异")
            return True
        else:
            print("❌ LoRA微调效果不明显")
            return False
            
    except Exception as e:
        print(f"✗ 对比失败: {e}")
        return False

if __name__ == "__main__":
    print("🔧 开始提取和测试LoRA权重...")
    
    # 1. 提取LoRA权重
    success, lora_dir = extract_lora_weights()
    
    if not success:
        print("❌ LoRA权重提取失败")
        exit(1)
    
    print(f"\n✅ LoRA权重提取成功！保存在: {lora_dir}")
    
    # 2. 测试提取的LoRA
    print("\n" + "="*70)
    test_success = test_extracted_lora()
    
    if test_success:
        print("✅ LoRA权重测试成功！")
        
        # 3. 效果对比
        compare_success = compare_with_original()
        
        if compare_success:
            print("\n🎉 LoRA微调效果验证成功！")
            print("查看对比结果:")
            print("  - lora_final_comparison/original.png: 原始模型结果")
            print("  - lora_final_comparison/lora.png: LoRA模型结果")
        else:
            print("\n⚠️  LoRA权重提取成功，但效果不明显")
    else:
        print("❌ LoRA权重测试失败") 