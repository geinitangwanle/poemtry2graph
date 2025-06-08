from diffusers import StableDiffusionPipeline
import torch

# --- 1. 指定 .safetensors 文件路径 ---
model_path = "shanshui_style.safetensors" # <-- 模型的实际路径

# --- 2. 使用 from_single_file 方法加载模型 ---
# 这是专门为加载单文件模型设计的便捷方法
pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16,local_files_only=True)
pipe.to("mps")

# --- 3. 编写提示词 ---
# 别忘了你的触发词！
prompt = "算鸟蒜鸟都不容易, a painting in tong tong tong sahur style"
negative_prompt = ""

# --- 4. 生成并保存图片 ---
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("dreambooth.png")

print("图片生成成功！")