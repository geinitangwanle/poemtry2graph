import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 1. 设置模型和设备
# 使用一个基础的 Stable Diffusion v1.5 模型
model_id_or_path = "models/diffusions"
# 如果有可用的 CUDA GPU，则使用它，否则使用 CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. 加载 img2img Pipeline
# torch_dtype=torch.float16 在 GPU 上能显著节省显存并加快速度
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    torch_dtype=torch.float16
).to(device)

# 3. 加载并准备输入图片
# 确保你有一个名为 'your_image.png' 的图片文件在脚本旁边
try:
    init_image = Image.open("/Users/tangren/Documents/poemtry2graph/lora_poem_output2000/05_千里莺啼绿映红，.png").convert("RGB")
except FileNotFoundError:
    print("错误: 图片未找到。请在该目录下放置一张图片。")
    # 创建一个512x512的灰色占位图以便脚本可以继续运行
    init_image = Image.new('RGB', (512, 512), color = 'gray')


# 确保输入图片是 512x512
init_image = init_image.resize((512, 512))

# 4. 设置风格转换的参数
prompt = "a vast landscape with distant mountains, a river winding through a village, small boats on the water, colorful blossoms on trees, birds in flight, a rustic house with a flag fluttering in the wind, serene and vivid" # 目标风格描述，与之前一致
negative_prompt = "blurry, low quality, deformed" # 不希望出现的内容
strength = 0.75 # 风格转换的强度 (0.0 - 1.0)。值越高，风格越强，但与原图差异越大。
guidance_scale = 7.5 # 提示词引导强度。值越高，图片越接近你的 prompt 描述。

# 5. 运行 Pipeline 并生成图片
# 使用 torch.Generator 来确保结果的可复现性
generator = torch.Generator(device=device).manual_seed(1024)
output_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=strength,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# 6. 保存结果
output_image.save("style_transferred_image.png")

print("风格转换完成！图片已保存为 'style_transferred_image.png'")