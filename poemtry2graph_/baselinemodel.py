from diffusers import StableDiffusionPipeline
import torch

# 指定模型路径
model_path = "models/diffusions"  # 基线模型路径

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

# 将模型移至GPU
device = "mps"
pipe = pipe.to(device)

# 生成图片的参数
prompt = "A vast snowy landscape with towering ice-covered mountains and frozen rivers, dotted with evergreen trees covered in white snow. The clear blue sky casts soft sunlight over the blocky terrain, where occasional wooden houses with smoking chimneys stand against the cold."  # 提示词
negative_prompt = ""  # 负面提示词

# 生成图片
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# 保存生成的图片
image.save("baseline.png")

print("图片生成完成!")

