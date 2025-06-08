import os
import requests
import csv
import torch
from diffusers import StableDiffusionPipeline

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-164ad8ec739c466aa7a53489f3f9eaaa"

# 模型路径配置
BASE_MODEL_PATH = "models/diffusions"
LORA_PATH = "joint_lora/5000"
POEM_FILE = "poems.txt"
OUTPUT_DIR = "lora_poem_output5000"

def translate_poem_with_deepseek(poem_text: str) -> str:
    """使用DeepSeek API翻译古诗"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """## 角色与目标
你是一位专业的AI助手，专门负责创作艺术提示词。你的特定角色是**"诗歌到提示词转换器"**。你的目标是读取一首中国古诗，并将其核心的视觉和情感精髓，转换成一段简洁、有效、适合文生图AI模型（如Stable Diffusion）使用的英文提示词。

## 最重要规则：核心词汇库
你将要为其生成提示词的AI模型，是在一个有限且特定的词汇库上训练的。为了生成质量最好、风格最统一的图像，你必须主要使用下方**"核心词汇库"**中列出的单词和短语来构建你的提示词。这是你的首要约束。如果词汇库中有简单的词，就不要创造复杂的新词。

--- 核心词汇库 (CORE VOCABULARY PALETTE) ---

1. 主体与物品 (Subjects & Objects):
人物 (People): a man, a woman, a scholar, a teacher, an archer, a figure, figures, a lone figure, pupils, a bearded man  
动物 (Animals): a horse, horses, a tiger, two cats, a rat, a bird, birds, a kingfisher, two cranes, swans, camels, a small dog  
植物 (Plants): a tree, trees, a pine tree, a weeping willow tree, bamboo, a lotus flower, lotus leaves, blossoms, foliage, reeds, red leaves  
建筑 (Structures): a pavilion, a house, a cottage, a bridge, a balcony, a terrace, a path, a stone path, a wall, a railing  
物品 (Items): a boat, small boats, a mirror, a scroll, a sword, a staff, a tea set, an incense burner, a bow and arrow, a carrying pole  

2. 场景与环境 (Scenery & Environment):
地貌 (Landforms): a landscape, a mountain, mountains, mountain peaks, a cliff, a cliffside, a valley, a riverbank, a shore, an island, rocks
水体 (Water): a river, a lake, a waterfall, a stream, mist, fog, a sea of clouds  
氛围 (Atmosphere): a vast landscape, a sparse background, a plain background, a monochromatic landscape, a quiet landscape, distant hills

3. 特质、情绪与风格 (Qualities, Moods, & Styles):
情绪 (Mood): serene, tranquil, quiet, elegant, graceful, dramatic, powerful, dynamic, rustic, minimalist
视觉 (Visuals): expressive, flowing, vivid, colorful, soft, hazy, misty, aged, sepia-toned, monochrome, energetic brushstrokes

4. 动作与姿态 (Actions & Poses):
sitting, standing, walking, running, galloping, flying, in mid-flight, sailing, playing an instrument, reading, writing, gazing, looking up, at full draw, aiming, perched on a branch

--- 词汇库结束 ---

## 需遵循的流程
1. 分析诗歌：仔细阅读输入的诗歌。识别出其中的主要主体（人物, 动物）、场景（场景）、关键物品（物品）以及整体的意境（意境）。
2. 映射到词汇库：对于你识别出的每一个元素，从**"核心词汇库"**中找到最贴切、最合适的词或短语。
3. 构建提示词：将选定的关键词组合成一个连贯的英文段落。使用逗号分隔不同的概念。从主要主体开始，然后描述环境和氛围。
4. 聚焦视觉描述：提示词应该是纯粹的对潜在画面的描述。最终的提示词中不要包含你自己的解读、诗歌的标题或中文原文。
5. 重要约束：不要在提示词中包含任何具体的艺术风格词汇（如水墨画、国画等），只描述场景内容。

现在你已准备就绪。我将向你提供一首诗，请根据以上所有规则，生成完美的提示词。"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "李白《静夜思》：床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
            {"role": "assistant", "content": "a lone figure, a scholar, sitting in a minimalist room, gazing up, tranquil, quiet mood"},
            {"role": "user", "content": "王之涣《登鹳雀楼》：白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
            {"role": "assistant", "content": "a vast mountain landscape, a pavilion on a cliffside, a wide river flowing into the distance, serene, a lone figure gazing"},
            {"role": "user", "content": "杜甫《春望》：国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。"},
            {"role": "assistant", "content": "a lone figure gazing over a landscape, distant mountains and a river, an aged wall with overgrown foliage and blossoms, birds in a tree, quiet and tranquil mood"},
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
            return f"a lone figure, a landscape, mountains, a river, tranquil, quiet"
    except Exception as e:
        print(f"API调用失败：{e}")
        return f"a landscape, mountains, a river, serene, tranquil"

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
    
    print("正在加载联合LoRA权重...")
    pipe.load_lora_weights(LORA_PATH)
    pipe = pipe.to(device)
    
    # 设置LoRA适配器
    adapter_names = list(pipe.get_active_adapters())
    if adapter_names:
        adapter_name = adapter_names[0]
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        print(f"设置联合LoRA适配器: {adapter_name}")
    else:
        print("警告: 没有找到活跃的LoRA适配器")
        return
    
    # 读取古诗
    try:
        with open(POEM_FILE, 'r', encoding='utf-8') as f:
            poems = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 找不到古诗文件 {POEM_FILE}")
        print("请创建一个 poems.txt 文件，每行一首古诗")
        return
    
    # 创建日志文件
    log_file = open(os.path.join(OUTPUT_DIR, "generation_log.csv"), "w", newline='', encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["序号", "原诗", "英文提示词", "图片路径"])
    
    print(f"\n开始处理 {len(poems)} 首古诗...")
    print("=" * 60)
    
    for idx, poem in enumerate(poems, 1):
        try:
            print(f"\n[{idx:02d}] 原诗：{poem}")
            
            # 使用DeepSeek翻译
            english_prompt = translate_poem_with_deepseek(poem)
            print(f"英文提示词：{english_prompt}")
            
            # 添加风格前缀
            full_prompt = f"{english_prompt}"
            
            # 生成图像
            print("正在生成图像...")
            image = pipe(
                prompt=full_prompt,
                negative_prompt="",
                num_inference_steps=40,
                guidance_scale=7.5,
                generator=torch.Generator(device=device).manual_seed(42 + idx)
            ).images[0]
            
            # 保存图像
            filename = f"{idx:02d}_{poem[:8].replace(' ', '_')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            image.save(filepath)
            
            # 记录日志
            writer.writerow([idx, poem, english_prompt, filepath])
            
            print(f"✅ 生成完成：{filename}")
            
        except Exception as e:
            print(f"❌ 第 {idx} 首处理失败：{e}")
            continue
    
    log_file.close()
    
    print("\n" + "=" * 60)
    print(f"🎉 批量生成完毕！图片保存在 {OUTPUT_DIR} 目录")
    print("=" * 60)

if __name__ == "__main__":
    main() 