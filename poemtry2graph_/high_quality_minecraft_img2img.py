import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import time

class MinecraftStyleConverter:
    def __init__(self, model_path="models/diffusions", lora_path="joint_lora_mc/2000"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🖥️  使用设备: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载基础模型和LoRA权重"""
        print("📥 正在加载基础模型...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        print("🎯 正在加载Minecraft风格LoRA权重...")
        self.pipe.load_lora_weights(self.lora_path)
        
        # 设置LoRA适配器
        adapter_names = list(self.pipe.get_active_adapters())
        if adapter_names:
            adapter_name = adapter_names[0]
            self.pipe.set_adapters([adapter_name], adapter_weights=[1.0])
            print(f"✅ 设置Minecraft风格LoRA适配器: {adapter_name}")
        else:
            print("⚠️  警告: 没有找到活跃的LoRA适配器")
    
    def get_quality_settings(self, quality_level="high"):
        """获取不同质量等级的参数设置"""
        settings = {
            "draft": {
                "size": (512, 512),
                "steps": 20,
                "guidance_scale": 7.0,
                "strength": 0.7,
                "description": "快速预览质量"
            },
            "medium": {
                "size": (768, 768),
                "steps": 30,
                "guidance_scale": 7.5,
                "strength": 0.6,
                "description": "中等质量"
            },
            "high": {
                "size": (768, 768),
                "steps": 50,
                "guidance_scale": 8.0,
                "strength": 0.6,
                "description": "高质量"
            },
            "ultra": {
                "size": (1024, 1024),
                "steps": 80,
                "guidance_scale": 8.5,
                "strength": 0.55,
                "description": "超高质量（需要更多时间和显存）"
            }
        }
        return settings.get(quality_level, settings["high"])
    
    def convert_image(self, input_image_path, custom_prompt=None, quality_level="high", seed=1024):
        """转换图像为Minecraft风格"""
        
        # 获取质量设置
        settings = self.get_quality_settings(quality_level)
        print(f"🎨 使用质量等级: {quality_level} - {settings['description']}")
        print(f"📐 图片尺寸: {settings['size']}")
        print(f"🔢 推理步数: {settings['steps']}")
        
        # 加载输入图片
        try:
            init_image = Image.open(input_image_path).convert("RGB")
            print(f"📸 成功加载输入图片: {input_image_path}")
        except FileNotFoundError:
            print(f"❌ 错误: 找不到图片 {input_image_path}")
            return None
        
        # 调整图片尺寸
        init_image = init_image.resize(settings['size'])
        
        # 设置提示词
        if custom_prompt:
            minecraft_prompt = custom_prompt
        else:
            minecraft_prompt = """A beautiful minecraft style landscape with wooden houses nestled in rolling hills, 
            dense evergreen forest, calm flowing rivers, clear blue sky, blocky pixelated appearance, 
            stylized cubic design, peaceful village, cherry blossom trees, glowing lanterns"""
        
        # 优化的负面提示词
        negative_prompt = """blurry, low quality, distorted, bad anatomy, watermark, text, realistic photography, 
        smooth textures, high detail realistic, non-blocky, overly detailed, noise, artifacts, 
        deformed structures, asymmetrical, poor composition"""
        
        print(f"💭 Minecraft提示词: {minecraft_prompt[:80]}...")
        
        # 生成图片
        print("🎮 正在生成Minecraft风格图片...")
        start_time = time.time()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        output_image = self.pipe(
            prompt=minecraft_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=settings['strength'],
            guidance_scale=settings['guidance_scale'],
            num_inference_steps=settings['steps'],
            generator=generator
        ).images[0]
        
        end_time = time.time()
        print(f"⏱️  生成完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        output_filename = f"{base_name}_minecraft_{quality_level}_quality.png"
        output_image.save(output_filename)
        print(f"💾 图片已保存为: {output_filename}")
        
        # 创建对比图
        self._create_comparison(init_image, output_image, base_name, quality_level)
        
        return output_image
    
    def _create_comparison(self, original, converted, base_name, quality):
        """创建原图与转换图的对比"""
        width, height = original.size
        comparison_width = width * 2 + 30
        comparison_height = height + 60
        
        comparison_image = Image.new('RGB', (comparison_width, comparison_height), color='white')
        
        # 粘贴图片
        comparison_image.paste(original, (10, 30))
        comparison_image.paste(converted, (width + 20, 30))
        
        # 保存对比图
        comparison_filename = f"{base_name}_minecraft_comparison_{quality}.png"
        comparison_image.save(comparison_filename)
        print(f"📊 对比图已保存为: {comparison_filename}")

def main():
    # 初始化转换器
    converter = MinecraftStyleConverter()
    
    # 设置参数
    input_image = "joint_lora_output.png"  # 输入图片路径
    quality_levels = ["draft", "medium", "high", "ultra"]
    
    print("🎯 可用的质量等级:")
    for level in quality_levels:
        settings = converter.get_quality_settings(level)
        print(f"  {level}: {settings['description']} - {settings['size']}, {settings['steps']} steps")
    
    # 选择质量等级（您可以修改这里）
    selected_quality = "high"  # 可以改为 "draft", "medium", "high", "ultra"
    
    # 自定义提示词（可选）
    custom_prompt = None  # 如果想使用自定义提示词，在这里修改
    
    # 执行转换
    print(f"\n🚀 开始使用 {selected_quality} 质量等级进行转换...")
    result = converter.convert_image(
        input_image_path=input_image,
        custom_prompt=custom_prompt,
        quality_level=selected_quality,
        seed=1024
    )
    
    if result:
        print("🎉 Minecraft风格转换完成!")
    else:
        print("❌ 转换失败!")

if __name__ == "__main__":
    main() 