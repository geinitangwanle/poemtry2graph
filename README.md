# 诗意图生（Poemtry2Graph）

> 🌸 将中国古诗词转化为视觉艺术的AI项目 🎨

一个基于Stable Diffusion和LoRA微调技术的创新项目，能够将中国古诗词转换为对应的艺术图像。项目支持多种艺术风格，包括传统国画风格和现代Minecraft风格，让诗词之美在视觉世界中得到新的诠释。

## ✨ 项目特色

- 🎯 **智能诗词理解**：使用DeepSeek API智能解析古诗词的意境和视觉元素
- 🎨 **多种艺术风格**：支持传统诗意风格和现代Minecraft像素风格
- 🔧 **LoRA微调技术**：采用先进的LoRA（Low-Rank Adaptation）技术进行模型微调
- 🚀 **高效训练**：针对NVIDIA A800GPU优化，支持混合精度训练
- 📊 **完整工作流**：从数据处理到模型训练再到图像生成的完整pipeline

## 🏗️ 项目架构

```
poemtry2graph/
├── 📁 models/                 # 模型文件存储
├── 📁 contents/              # 内容数据
├── 📁 notebooks/             # Jupyter演示笔记本
├── 📁 utils/                 # 工具函数库
├── 📁 image_process/         # 图像处理工具
├── 📁 text_encoder/          # 文本编码器权重
├── 📁 poem_style/            # 诗意风格LoRA权重
├── 📁 mc_style/              # Minecraft风格LoRA权重
├── 📁 joint_lora/            # 联合训练LoRA权重
├── 🎨 poem_style.py          # 传统诗意风格生成
├── 🎮 mc_style.py            # Minecraft风格生成
├── 🔧 unet_text_encoder_lora_finetune.py  # 联合LoRA微调
├── 📊 baselinemodel.py       # 基线模型
├── 🖼️ image2image.py         # 图像到图像转换
└── 📝 poems.txt              # 诗词数据集
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（推荐）
- 16GB+ GPU显存（推荐A800/A100）

### 安装依赖

```bash
# 克隆项目
git clone <https://github.com/geinitangwanle/poemtry2graph.git>
cd poemtry2graph

# 安装依赖包
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install peft datasets pillow pandas tqdm
pip install requests openai  # DeepSeek API
```

### 配置API密钥

在使用前需要配置DeepSeek API密钥（用于智能诗词解析）：

```python
# 在 poem_style.py 和 mc_style.py 中修改
DEEPSEEK_API_KEY = "your-api-key-here"
```

## 🎨 主要功能

### 1. 传统诗意风格生成

使用传统中国画风格将古诗词转换为图像：

```bash
python poem_style.py
```

**特点：**
- 采用精心设计的视觉词汇库
- 支持山水、人物、花鸟等传统题材
- 优雅的水墨画效果

### 2. Minecraft像素风格生成

将古诗词转换为充满创意的Minecraft风格场景：

```bash
python mc_style.py
```

**特点：**
- 方块化、像素化的独特美学
- 包含Minecraft世界的建筑和生物元素
- 现代与传统的完美融合

### 3. LoRA模型微调

使用自定义数据集训练专用的LoRA适配器：

```bash
# UNet + Text Encoder联合微调
python unet_text_encoder_lora_finetune.py \
    --model_id ./models/diffusions \
    --csv_path ./data/image_prompts.csv \
    --image_dir ./data/images/ \
    --output_dir ./output_lora \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 4. 图像到图像转换

基于现有图像进行风格转换：

```bash
python image2image.py
```

## 📊 数据格式

### 诗词数据格式（poems.txt）
```
空山新雨后，天气晚来秋
明月松间照，清泉石上流
春眠不觉晓，处处闻啼鸟
...
```

### 训练数据格式（CSV）
```csv
image_id	poem	description
001	空山新雨后，天气晚来秋	山间雨后清新景色...
002	明月松间照，清泉石上流	月光下的松林溪流...
```

## 🔧 高级配置

### LoRA训练参数调整

```python
# LoRA配置参数
lora_rank = 16          # LoRA秩，影响适配器容量
lora_alpha = 32         # LoRA alpha，控制缩放因子
lora_dropout = 0.1      # Dropout率，防止过拟合
learning_rate = 1e-4    # UNet学习率
text_encoder_lr = 5e-5  # Text Encoder学习率
```

### 生成参数优化

```python
# 图像生成参数
num_inference_steps = 50    # 推理步数，影响质量
guidance_scale = 7.5        # 引导强度，控制遵循prompt程度
resolution = 768            # 输出分辨率
```

## 📈 性能优化

### GPU内存优化
- 支持梯度累积减少显存占用
- 混合精度训练（FP16）
- 模型编译加速（torch.compile）

### 训练加速
- 多GPU分布式训练
- 数据并行处理
- 高效的数据加载器

## 🎯 应用场景

1. **教育领域**：诗词教学可视化，帮助学生理解古诗意境
2. **文化传播**：传统文化的现代化展示
3. **艺术创作**：AI辅助的诗意艺术创作
4. **游戏开发**：基于诗词的场景和关卡设计
5. **数字人文**：古典文学的数字化研究

## 🛠️ 开发工具

### Jupyter Notebooks
- `demo.ipynb`：基础演示
- `demoTrain.ipynb`：训练演示
- `example.ipynb`：完整示例

### 实用工具
- `utils/joint_lora.py`：LoRA权重合并
- `utils/process_lora_weights.py`：权重处理
- `utils/final_comparison.py`：效果对比分析

## 📋 TODO

- [ ] 支持更多艺术风格（油画、素描等）
- [ ] 添加Web界面和API服务
- [ ] 优化模型推理速度
- [ ] 扩展多语言诗词支持
- [ ] 增加用户自定义风格训练

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请遵循以下指南：

1. Fork项目并创建feature分支
2. 确保代码符合PEP8规范
3. 添加适当的测试和文档
4. 提交PR并详细描述修改内容


## 🙏 致谢

- Stable Diffusion团队提供的基础模型
- DeepSeek提供的智能API服务
- 开源社区的diffusers和transformers库
- 所有贡献者和使用者的支持


---

*让古诗词在AI的画笔下绽放新的光彩* ✨ 