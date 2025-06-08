import os
import requests
import csv
from diffusers import StableDiffusionPipeline
import torch

DEEPSEEK_API_KEY = ""

# 创建输出文件夹
os.makedirs("output", exist_ok=True)

# Stable Diffusion 模型加载
pipe = StableDiffusionPipeline.from_pretrained(
    "./notebooks/models/diffusion",
    torch_dtype=torch.float16
).to("mps")

# 调用 DeepSeek 翻译古诗
def translate_poem(poem_text: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个翻译家，请将一首古诗翻译成简洁的、不超过60个英文单词的句子，用于生成图像，具有画面感与意境。"},
            {"role": "user", "content": poem_text}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"翻译失败：{response.status_code}\n{response.text}")

# 主批量生成逻辑
def generate_images_from_file(poem_file_path="poems.txt"):
    with open(poem_file_path, 'r', encoding='utf-8') as f:
        poems = [line.strip() for line in f if line.strip()]

    log_file = open("poem_log.csv", "w", newline='', encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["Index", "Poem", "Prompt", "ImagePath"])

    for idx, poem in enumerate(poems, 1):
        try:
            print(f"\n[{idx:03}] 原文：{poem}")
            english = translate_poem(poem)
            full_prompt = f"A  painting, depicting: {english}"
            print(f"Prompt: {full_prompt}")

            image = pipe(full_prompt).images[0]
            filename = f"output/{idx:03}.png"
            image.save(filename)

            writer.writerow([idx, poem, full_prompt, filename])
            print(f"✅ 图像已保存至：{filename}")

        except Exception as e:
            print(f"❌ 第 {idx} 首失败：{e}")

    log_file.close()
    print("\n🎉 所有古诗处理完毕！")

# 执行批处理
if __name__ == "__main__":
    generate_images_from_file()