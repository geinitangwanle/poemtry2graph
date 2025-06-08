import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# 1. 设置模型和设备
# 使用基础的 Stable Diffusion v1.5 模型
model_id_or_path = "models/diffusions"
# LoRA模型路径 - 您的Minecraft风格微调模型
lora_path = "joint_lora_mc/2000"
# 如果有可用的GPU，则使用它，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 2. 加载 img2img Pipeline
print("正在加载基础模型...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# 3. 加载您的Minecraft风格LoRA权重
print("正在加载Minecraft风格LoRA权重...")
pipe.load_lora_weights(lora_path)

# 设置LoRA适配器
adapter_names = list(pipe.get_active_adapters())
if adapter_names:
    adapter_name = adapter_names[0]
    pipe.set_adapters([adapter_name], adapter_weights=[1.0])
    print(f"设置Minecraft风格LoRA适配器: {adapter_name}")
else:
    print("警告: 没有找到活跃的LoRA适配器")

# 4. 加载并准备输入图片
try:
    init_image = Image.open("/Users/tangren/Documents/poemtry2graph/lora_poem_output2000/07_白日依山尽，黄河.png").convert("RGB")
    print("成功加载输入图片")
except FileNotFoundError:
    print("错误: 图片未找到。请在该目录下放置一张图片。")
    # 创建一个512x512的灰色占位图以便脚本可以继续运行
    init_image = Image.new('RGB', (512, 512), color='gray')

# 确保输入图片是合适的尺寸（推荐512x512或768x768）
target_size = (768, 768)  # 使用更高分辨率获得更好质量
init_image = init_image.resize(target_size)
print(f"图片尺寸调整为: {target_size}")

# 5. 设置Minecraft风格转换的参数
# Minecraft风格提示词
minecraft_prompt = "A dramatic sunset over rolling hills with a wide river flowing through the valley, leading to distant mountain ranges under a bright sky with soft clouds. The landscape features blocky stone formations and scattered evergreen trees along the riverbanks."

# 负面提示词 - 避免不希望的元素
negative_prompt = ""

# 参数设置 - 针对LoRA模型优化
strength = 0.6  # 风格转换强度 (0.0-1.0)，0.6是比较好的平衡点
guidance_scale = 8.0  # 提示词引导强度，LoRA模型建议稍高一些
num_inference_steps = 40  # 推理步数，更多步数 = 更好质量

print(f"Minecraft提示词: {minecraft_prompt}")
print(f"参数设置 - strength: {strength}, guidance_scale: {guidance_scale}, steps: {num_inference_steps}")

# 6. 运行 Pipeline 并生成图片
print("正在生成Minecraft风格图片...")
generator = torch.Generator(device=device).manual_seed(1024)

output_image = pipe(
    prompt=minecraft_prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator
).images[0]

# 7. 保存结果
output_filename = "minecraft_style_converted.png"
output_image.save(output_filename)

print(f"✅ Minecraft风格转换完成！图片已保存为 '{output_filename}'")

# 可选：保存对比图
if os.path.exists("joint_lora_output.png"):
    # 创建对比图
    comparison_width = target_size[0] * 2 + 20
    comparison_height = target_size[1] + 40
    comparison_image = Image.new('RGB', (comparison_width, comparison_height), color='white')
    
    # 粘贴原图和转换后的图
    comparison_image.paste(init_image, (0, 20))
    comparison_image.paste(output_image, (target_size[0] + 20, 20))
    
    comparison_image.save("minecraft_conversion_comparison.png")
    print("📊 对比图已保存为 'minecraft_conversion_comparison.png'")