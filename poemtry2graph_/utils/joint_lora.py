from diffusers import StableDiffusionPipeline
import torch

# 指定模型路径
base_model_path = "models/diffusions"  # 基线模型路径
lora_path = "joint_lora/final"  # 联合LoRA权重路径

# 加载基础模型
print("正在加载基础模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# 加载联合训练的LoRA权重（包含UNet和Text Encoder）
print("正在加载联合LoRA权重...")
pipe.load_lora_weights(lora_path)

# 将模型移至设备
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")
pipe = pipe.to(device)

# 生成图片的参数
prompt = "A seated man with a beard, wearing a cap and holding a gnarled staff, looks up at the red leaves of the large tree branch above him."  # 提示词
negative_prompt = ""  # 负面提示词
unet_lora_scale = 1.0  # UNet LoRA权重强度 (0.0-1.0)
text_encoder_lora_scale = 1.0  # Text Encoder LoRA权重强度 (0.0-1.0)

# 设置LoRA适配器
adapter_names = list(pipe.get_active_adapters())
if adapter_names:
    adapter_name = adapter_names[0]
    pipe.set_adapters([adapter_name], adapter_weights=[unet_lora_scale])
    print(f"设置联合LoRA适配器: {adapter_name}")
    print(f"UNet LoRA强度: {unet_lora_scale}")
    print(f"Text Encoder LoRA强度: {text_encoder_lora_scale}")
else:
    print("警告: 没有找到活跃的LoRA适配器")

# 生成图片
print("开始生成图片...")
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# 保存生成的图片
image.save("joint_lora_output.png")
