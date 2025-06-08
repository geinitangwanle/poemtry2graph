#!/usr/bin/env python3
"""
最终的LoRA微调效果对比测试
"""

import torch
from diffusers import StableDiffusionPipeline
import os
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

def final_comparison_test():
    """进行原始模型vs LoRA模型对比测试"""
    
    print("="*80)
    print("LoRA微调效果验证")
    print("="*80)
    
    BASE_MODEL_PATH = "./models/diffusions"
    LORA_PATH = "./fixed_lora"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 诗意测试提示词
    test_prompts = [
        "举头望明月，低头思故乡",
        "春江花月夜，诗意山水画",
        "古典中国山水，水墨画风格，唐诗意境",
        "月下江景，古诗词意境，中国传统绘画",
        "山水诗意，云雾缭绕，水墨淡彩"
    ]
    
    print(f"设备: {DEVICE}")
    print(f"测试提示词数量: {len(test_prompts)}")
    print(f"LoRA路径: {LORA_PATH}")
    
    try:
        # 1. 原始模型批量生成
        print("\n1️⃣ 原始模型批量生成...")
        original_pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to(DEVICE)
        
        os.makedirs("final_comparison", exist_ok=True)
        original_images = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"  生成原始图像 {i+1}/{len(test_prompts)}: {prompt[:30]}...")
            
            with torch.no_grad():
                image = original_pipeline(
                    prompt=prompt,
                    num_inference_steps=25,  # 增加步数以获得更好质量
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator(device=DEVICE).manual_seed(42)
                ).images[0]
            
            image_path = f"final_comparison/original_{i+1}.png"
            image.save(image_path)
            original_images.append(np.array(image))
            print(f"    ✓ 已保存: {image_path}")
        
        print("✅ 原始模型生成完成")
        
        # 2. LoRA模型批量生成
        print("\n2️⃣ LoRA模型批量生成...")
        lora_pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        
        # 加载LoRA权重
        lora_pipeline.load_lora_weights(LORA_PATH)
        lora_pipeline = lora_pipeline.to(DEVICE)
        
        lora_images = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"  生成LoRA图像 {i+1}/{len(test_prompts)}: {prompt[:30]}...")
            
            with torch.no_grad():
                image = lora_pipeline(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator(device=DEVICE).manual_seed(42)
                ).images[0]
            
            image_path = f"final_comparison/lora_{i+1}.png"
            image.save(image_path)
            lora_images.append(np.array(image))
            print(f"    ✓ 已保存: {image_path}")
        
        print("✅ LoRA模型生成完成")
        
        # 3. 计算差异分析
        print("\n3️⃣ 差异分析...")
        total_diff = 0
        significant_diffs = 0
        
        for i in range(len(test_prompts)):
            orig_img = original_images[i]
            lora_img = lora_images[i]
            
            # 计算像素差异
            pixel_diff = np.mean(np.abs(orig_img.astype(float) - lora_img.astype(float)))
            total_diff += pixel_diff
            
            if pixel_diff > 10:  # 如果差异大于阈值
                significant_diffs += 1
            
            print(f"  图像 {i+1} 差异: {pixel_diff:.2f}")
        
        avg_diff = total_diff / len(test_prompts)
        print(f"\n📊 统计结果:")
        print(f"  平均像素差异: {avg_diff:.2f}")
        print(f"  显著差异图像: {significant_diffs}/{len(test_prompts)}")
        print(f"  差异比例: {100 * significant_diffs / len(test_prompts):.1f}%")
        
        # 4. 效果评估
        print(f"\n4️⃣ 效果评估:")
        
        if avg_diff > 15:
            print("🎉 微调效果显著！")
            print("   - LoRA模型与原始模型生成的图像有明显差异")
            print("   - 微调成功改变了模型的生成风格")
            success = True
        elif avg_diff > 5:
            print("✅ 微调效果中等")
            print("   - LoRA模型有一定的风格变化")
            print("   - 可能需要调整参数以获得更明显的效果")
            success = True
        else:
            print("⚠️  微调效果微弱")
            print("   - LoRA模型与原始模型差异很小")
            print("   - 建议检查训练参数和数据")
            success = False
        
        # 5. 生成对比图
        print(f"\n5️⃣ 生成对比图...")
        create_comparison_grid(test_prompts)
        
        return success, avg_diff, significant_diffs
        
    except Exception as e:
        print(f"✗ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def create_comparison_grid(prompts):
    """创建对比网格图"""
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        print("  创建对比网格...")
        
        # 加载图像
        grid_images = []
        for i in range(len(prompts)):
            orig_img = Image.open(f"final_comparison/original_{i+1}.png")
            lora_img = Image.open(f"final_comparison/lora_{i+1}.png")
            
            # 创建标题
            img_width = orig_img.width
            title_height = 30
            
            # 创建带标题的图像
            orig_with_title = Image.new('RGB', (img_width, orig_img.height + title_height), 'white')
            lora_with_title = Image.new('RGB', (img_width, lora_img.height + title_height), 'white')
            
            # 添加图像
            orig_with_title.paste(orig_img, (0, title_height))
            lora_with_title.paste(lora_img, (0, title_height))
            
            # 添加文字（如果可能的话）
            try:
                draw_orig = ImageDraw.Draw(orig_with_title)
                draw_lora = ImageDraw.Draw(lora_with_title)
                draw_orig.text((10, 5), "Original", fill='black')
                draw_lora.text((10, 5), "LoRA", fill='black')
            except:
                pass  # 如果字体不可用就跳过
            
            grid_images.append((orig_with_title, lora_with_title))
        
        # 创建网格
        if grid_images:
            img_width = grid_images[0][0].width
            img_height = grid_images[0][0].height
            
            # 2列网格 (原始 | LoRA)
            grid_width = img_width * 2
            grid_height = img_height * len(prompts)
            
            grid = Image.new('RGB', (grid_width, grid_height), 'white')
            
            for i, (orig_img, lora_img) in enumerate(grid_images):
                y_offset = i * img_height
                grid.paste(orig_img, (0, y_offset))
                grid.paste(lora_img, (img_width, y_offset))
            
            grid.save("final_comparison/comparison_grid.png")
            print("    ✓ 对比网格已保存: final_comparison/comparison_grid.png")
        
    except Exception as e:
        print(f"    ⚠️  网格创建失败: {e}")

if __name__ == "__main__":
    print("🚀 开始最终LoRA微调效果验证...")
    
    success, avg_diff, significant_diffs = final_comparison_test()
    
    print("\n" + "="*80)
    print("📋 最终报告")
    print("="*80)
    
    if success:
        print("🎉 LoRA微调验证成功！")
        print(f"📈 平均像素差异: {avg_diff:.2f}")
        print(f"📊 显著差异图像: {significant_diffs}/5")
        print("\n📁 生成的文件:")
        print("  📂 final_comparison/")
        print("    - original_1.png ~ original_5.png (原始模型生成)")
        print("    - lora_1.png ~ lora_5.png (LoRA模型生成)")
        print("    - comparison_grid.png (对比网格图)")
        print("\n💡 建议:")
        print("  1. 查看 comparison_grid.png 来直观比较效果")
        print("  2. 可以尝试不同的提示词来测试模型风格")
        print("  3. 调整 guidance_scale 和 num_inference_steps 优化效果")
    else:
        print("❌ LoRA微调效果不明显")
        print("💡 改进建议:")
        print("  1. 增加训练轮数 (epochs)")
        print("  2. 调整学习率")
        print("  3. 检查训练数据质量")
        print("  4. 调整LoRA参数 (rank, alpha)")
    
    print("\n🔥 LoRA微调项目完成！") 