import os
import requests
import csv
import torch
from diffusers import StableDiffusionPipeline

# DeepSeek API配置
DEEPSEEK_API_KEY = ""

# 模型路径配置
BASE_MODEL_PATH = "models/diffusions"
LORA_PATH = "joint_lora_mc/20000"  # Minecraft风格LoRA模型路径
OUTPUT_DIR = "minecraft_poem_output20000"
POEM_FILE = "poem.txt"  # 古诗文件路径

def translate_poem_with_deepseek(poem_text: str) -> str:
    """使用DeepSeek API将古诗转换为Minecraft风格的提示词"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """## 角色与目标
你是一位专业的AI助手，专门负责创作Minecraft风格的艺术提示词。你的特定角色是**"古诗到Minecraft场景转换器"**。你的目标是读取一首中国古诗，并将其核心的视觉和情感精髓，转换成一段简洁、有效、适合Minecraft风格图像生成的英文提示词。

## 最重要规则：Minecraft核心词汇库
你必须主要使用下方**"Minecraft核心词汇库"**中列出的单词和短语来构建你的提示词。这是你的首要约束。

--- Minecraft核心词汇库 (MINECRAFT VOCABULARY PALETTE) ---

1. 建筑与结构 (Buildings & Structures):
建筑类型: a wooden house, a stone castle, a cozy cottage, a tall tower, a fortified gate, a red barn, a treehouse, medieval-style village, farmland
建筑细节: steep roof, smoking chimney, wide porch, pointed blue roof, half-timbered house, arched bridges, stone buildings

2. 自然景观 (Natural Landscapes):
地形: grassy plains, rolling hills, mountain ranges, rocky cliffs, peaceful valley, hillside village, coastal forest
水体: a calm lake, a gentle river, a wide river, a waterfall, still waters, tranquil river, clear streams
植物: cherry blossom trees, pink-blossomed trees, dense forest, evergreen trees, tall trees, bamboo, wildflowers, blooming flowers

3. Minecraft生物 (Minecraft Creatures):
动物: cows, sheep, white woolly sheep, a pink pig, a white bird, a tiger, a camel, fish, a panda
人物: blocky figures, a blocky human-like figure, a bearded figure, two blocky figures

4. Minecraft环境特色 (Minecraft Environmental Features):
照明: glowing lights, lantern-lit paths, sunset light, golden sunset, soft sunlight, warm light filtering
方块化特征: blocky form, stylized design, pixelated appearance
游戏元素: coral reefs, giant red mushrooms, glowing vines, scattered tools and supplies

5. 氛围与情绪 (Atmosphere & Mood):
天空: clear blue sky, bright sky, dramatic sunset, soft clouds, blue sky with clouds
光线: soft evening light, afternoon light, midday sun, glowing, bathed in sunset light
情绪: peaceful, serene, cozy, tranquil, vibrant, warm, calm

--- 词汇库结束 ---

## 需遵循的流程
1. 分析诗歌：仔细阅读输入的诗歌，识别出其中的主要元素：人物、自然景观、建筑、动物、情绪氛围。
2. Minecraft转换：将诗歌元素转换为Minecraft世界中的对应场景。例如：
   - 山水 → rolling hills, mountain ranges, calm lake
   - 房屋村庄 → wooden house, cozy cottage, peaceful village
   - 人物 → blocky figures, a bearded figure
   - 动物 → cows, sheep, birds等
3. 构建场景：从**"Minecraft核心词汇库"**中选择词汇，构建一个连贯的Minecraft风格场景描述。
4. 保持方块化特色：确保描述体现Minecraft的方块化、像素化特征。
5. 重要约束：不要在提示词中包含任何其他艺术风格词汇，只描述Minecraft风格的场景内容。

## 示例转换：
古诗意境 → Minecraft场景描述的转换模式：
- 山居隐逸 → A cozy wooden house nestled in rolling hills, surrounded by dense forest and calm streams
- 江边渔翁 → A blocky figure stands beside a gentle river, with wooden houses on distant hills under a clear blue sky
- 春花烂漫 → A peaceful meadow filled with blooming cherry blossom trees and wildflowers, with soft sunlight filtering through
- 夜静月明 → A tranquil village with glowing lanterns under a starlit sky, surrounded by grassy plains

现在你已准备就绪。我将向你提供一首诗，请根据以上所有规则，生成完美的Minecraft风格提示词。"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "李白《静夜思》：床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
            {"role": "assistant", "content": "A blocky figure sits in a cozy wooden house, gazing up through a window at the bright moonlight. The peaceful village lies under a clear starlit sky, surrounded by grassy plains and distant hills."},
            {"role": "user", "content": "王之涣《登鹳雀楼》：白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
            {"role": "assistant", "content": "A tall stone tower rises above a vast landscape, overlooking a wide river flowing toward distant mountain ranges. The scene is bathed in golden sunset light with rolling hills stretching to the horizon."},
            {"role": "user", "content": "杜甫《春望》：国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。"},
            {"role": "assistant", "content": "A peaceful village with stone buildings and wooden houses sits among rolling hills, surrounded by dense forest and blooming wildflowers. Birds fly overhead while a gentle river flows through the valley."},
            {"role": "user", "content": poem_text}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"DeepSeek API错误：{response.status_code}")
            return f"A blocky figure stands in a peaceful minecraft landscape with rolling hills, wooden houses, and a calm river under a clear blue sky"
    except Exception as e:
        print(f"API调用失败：{e}")
        return f"A cozy village with wooden houses nestled in grassy hills, surrounded by dense forest and flowing streams"

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载基础模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    print("正在加载Minecraft风格LoRA权重...")
    pipe.load_lora_weights(LORA_PATH)
    pipe = pipe.to(device)
    
    # 设置LoRA适配器
    adapter_names = list(pipe.get_active_adapters())
    if adapter_names:
        adapter_name = adapter_names[0]
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        print(f"设置Minecraft风格LoRA适配器: {adapter_name}")
    else:
        print("警告: 没有找到活跃的LoRA适配器")
        return
    
    # 读取古诗
    try:
        with open(POEM_FILE, 'r', encoding='utf-8') as f:
            poems = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 找不到古诗文件 {POEM_FILE}")
        print("请创建一个 classic_poems.txt 文件，每行一首古诗")
        return
    
    # 创建日志文件
    log_file = open(os.path.join(OUTPUT_DIR, "minecraft_generation_log.csv"), "w", newline='', encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["序号", "原诗", "Minecraft英文提示词", "图片路径"])
    
    print(f"\n开始处理 {len(poems)} 首古诗（Minecraft风格）...")
    print("=" * 60)
    
    for idx, poem in enumerate(poems, 1):
        try:
            print(f"\n[{idx:02d}] 原诗：{poem}")
            
            # 使用DeepSeek转换为Minecraft风格提示词
            minecraft_prompt = translate_poem_with_deepseek(poem)
            print(f"Minecraft提示词：{minecraft_prompt}")
            
            # 生成图像 - 高质量设置
            print("正在生成Minecraft风格图像...")
            image = pipe(
                prompt=minecraft_prompt,
                negative_prompt="blurry, low quality, distorted, bad anatomy, watermark, text, realistic photography, smooth textures, high detail realistic, non-blocky, overly detailed, noise, artifacts, deformed structures, asymmetrical, poor composition",
                num_inference_steps=60,  # 增加推理步数以提高质量
                guidance_scale=8.5,  # 稍微提高引导强度
                width=768,  # 使用更高分辨率
                height=768,
                generator=torch.Generator(device=device).manual_seed(42 + idx)
            ).images[0]
            
            # 保存图像
            filename = f"minecraft_{idx:02d}_{poem[:8].replace(' ', '_')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            image.save(filepath)
            
            # 记录日志
            writer.writerow([idx, poem, minecraft_prompt, filepath])
            
            print(f"✅ Minecraft风格图像生成完成：{filename}")
            
        except Exception as e:
            print(f"❌ 第 {idx} 首处理失败：{e}")
            continue
    
    log_file.close()
    
    print("\n" + "=" * 60)
    print(f"🎉 Minecraft风格批量生成完毕！图片保存在 {OUTPUT_DIR} 目录")
    print("=" * 60)

if __name__ == "__main__":
    main() 