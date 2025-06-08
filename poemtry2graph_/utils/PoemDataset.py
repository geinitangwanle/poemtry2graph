from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import pandas as pd
import os

# --- 1. 定义核心参数 ---
MODEL_ID = "./models/diffusions" 
CSV_PATH = "./lora_data/POEM_IMAGE.csv"             # CSV文件路径
IMAGE_DIR = "./lora_data/processed_image/"                 # 图片文件夹路径
# --------------------
OUTPUT_DIR = "./lora_poem_style"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 1

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 创建自定义数据集 ---
class PoemDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer):
        """
        初始化数据集
        :param csv_path: CSV文件的路径
        :param image_dir: 存放所有图片的文件夹路径
        :param tokenizer: 用于文本编码的Tokenizer
        """
        # 1. 使用pandas读取CSV文件
        self.data = pd.read_csv(csv_path, sep='\t')
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer

        # 2. (优化) 预先扫描一次图片文件夹，建立 image_id -> file_path 的映射
        self.id_to_path = {
            p.stem: p for p in self.image_dir.iterdir() 
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
        }

        # 3. 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 将图片从 [0, 255] PIL Image 转换为 [0, 1] Tensor
            transforms.Normalize([0.5], [0.5]), # 将 [0, 1] 归一化到 [-1, 1]
        ])

    def __len__(self):
        # 数据集的总长度就是CSV文件的行数
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 根据索引从pandas DataFrame中获取一行数据
        row = self.data.iloc[idx]
        image_id = row['image_id']
        poem_text = row['poem']

        # 2. 从预先建立的映射中找到完整的图片路径
        img_path = self.id_to_path.get(image_id)
        if img_path is None:
            # 如果CSV中的id在图片文件夹里找不到，友好地跳过或报告
            print(f"Warning: Image file not found for ID: {image_id}, skipping.")
            # 返回None可以让DataLoader的collate_fn处理（需要自定义）或直接报错
            # 这里我们选择抛出错误，因为数据不匹配是严重问题
            raise FileNotFoundError(f"Image file not found for ID: {image_id} in {self.image_dir}")

        # 3. 读取并处理图像
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # 4. 读取并处理文本 (诗歌)
        input_ids = self.tokenizer(
            poem_text, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids.squeeze(0)}
    



