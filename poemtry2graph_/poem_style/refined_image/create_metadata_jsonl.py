#!/usr/bin/env python3
"""
CSV to metadata.jsonl Converter for Hugging Face LoRA Training
将CSV和图片目录转换为Hugging Face LoRA训练所需的metadata.jsonl格式

This script converts a CSV file containing image IDs and text captions,
along with a directory of images, into a metadata.jsonl file that can be
used with Hugging Face's official train_text_to_image_lora.py script.

Usage:
    python create_metadata_jsonl.py --csv_path my_poems.csv --image_dir images/ --output_file metadata.jsonl
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Convert CSV and image directory to metadata.jsonl for Hugging Face LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_metadata_jsonl.py --csv_path POEM_IMAGE.csv --image_dir processed_image/ --output_file metadata.jsonl
    python create_metadata_jsonl.py --csv_path my_data.csv --image_dir ./images --output_file ./output/metadata.jsonl
        """
    )
    
    parser.add_argument(
        "--csv_path", 
        type=str, 
        required=True,
        help="Path to the input CSV file containing image_id and poem columns"
    )
    
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing image files"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="metadata.jsonl",
        help="Path for the output metadata.jsonl file (default: metadata.jsonl)"
    )
    
    parser.add_argument(
        "--csv_separator",
        type=str,
        default="\t",
        help="CSV separator character (default: tab '\\t')"
    )
    
    parser.add_argument(
        "--image_id_column",
        type=str,
        default="image_id",
        help="Name of the image ID column in CSV (default: 'image_id')"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="poem",
        help="Name of the text/caption column in CSV (default: 'poem')"
    )
    
    return parser.parse_args()

def scan_image_directory(image_dir):
    """
    扫描图片目录，创建image_id到完整文件名的映射
    
    Args:
        image_dir (str): 图片目录路径
        
    Returns:
        dict: image_id -> 完整文件路径的映射字典
    """
    logging.info(f"正在扫描图片目录: {image_dir}")
    
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    
    # 支持的图片格式
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
    
    # 创建映射字典
    id_to_path = {}
    
    # 扫描目录中的所有文件
    for img_path in image_dir_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
            image_id = img_path.stem  # 文件名（不含扩展名）
            # 使用简单的路径拼接，确保兼容性
            relative_path = os.path.join(image_dir, img_path.name)
            id_to_path[image_id] = relative_path
    
    logging.info(f"找到 {len(id_to_path)} 个有效图片文件")
    logging.info(f"支持的格式: {', '.join(supported_extensions)}")
    
    if len(id_to_path) == 0:
        logging.warning("警告: 在指定目录中没有找到任何支持的图片文件")
    
    return id_to_path

def load_csv_data(csv_path, separator, image_id_column, text_column):
    """
    加载CSV数据
    
    Args:
        csv_path (str): CSV文件路径
        separator (str): CSV分隔符
        image_id_column (str): 图片ID列名
        text_column (str): 文本列名
        
    Returns:
        pandas.DataFrame: 加载的CSV数据
    """
    logging.info(f"正在加载CSV文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, sep=separator)
        logging.info(f"CSV文件加载成功，共 {len(df)} 行数据")
        
        # 检查必需的列是否存在
        if image_id_column not in df.columns:
            raise ValueError(f"CSV文件中缺少列: {image_id_column}")
        if text_column not in df.columns:
            raise ValueError(f"CSV文件中缺少列: {text_column}")
        
        logging.info(f"CSV列信息: {list(df.columns)}")
        
        # 显示前几行数据作为预览
        logging.info("CSV数据预览:")
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            logging.info(f"  行 {i+1}: {image_id_column}='{row[image_id_column]}', {text_column}='{str(row[text_column])[:50]}...'")
        
        return df
        
    except Exception as e:
        raise Exception(f"读取CSV文件时出错: {str(e)}")

def create_metadata_jsonl(df, id_to_path, output_file, image_id_column, text_column):
    """
    创建metadata.jsonl文件
    
    Args:
        df (pandas.DataFrame): CSV数据
        id_to_path (dict): image_id到文件路径的映射
        output_file (str): 输出文件路径
        image_id_column (str): 图片ID列名
        text_column (str): 文本列名
    """
    logging.info(f"开始创建metadata.jsonl文件: {output_file}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_entries = 0
    missing_images = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用tqdm显示进度条
        progress_bar = tqdm(
            df.iterrows(), 
            total=len(df),
            desc="处理CSV行",
            unit="行"
        )
        
        for idx, row in progress_bar:
            try:
                # 获取image_id和文本
                image_id = str(row[image_id_column]).strip()
                text = str(row[text_column]).strip()
                
                # 跳过空值
                if pd.isna(row[image_id_column]) or pd.isna(row[text_column]):
                    continue
                
                if not image_id or not text:
                    continue
                
                # 查找对应的图片文件
                if image_id in id_to_path:
                    # 创建JSON对象
                    json_obj = {
                        "file_name": id_to_path[image_id],
                        "text": text
                    }
                    
                    # 写入文件
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    valid_entries += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "有效": valid_entries,
                        "缺失": len(missing_images)
                    })
                else:
                    missing_images.append(image_id)
                    
            except Exception as e:
                logging.warning(f"处理第 {idx+1} 行时出错: {str(e)}")
                continue
    
    # 输出统计信息
    logging.info("=" * 50)
    logging.info("处理完成！统计信息:")
    logging.info(f"CSV总行数: {len(df)}")
    logging.info(f"找到的图片文件数: {len(id_to_path)}")
    logging.info(f"成功处理的条目数: {valid_entries}")
    logging.info(f"缺失图片的条目数: {len(missing_images)}")
    logging.info(f"输出文件: {output_file}")
    
    if missing_images:
        logging.warning(f"以下image_id没有找到对应的图片文件:")
        for missing_id in missing_images[:10]:  # 只显示前10个
            logging.warning(f"  - {missing_id}")
        if len(missing_images) > 10:
            logging.warning(f"  ... 还有 {len(missing_images) - 10} 个缺失的图片")
    
    logging.info("=" * 50)

def validate_output_file(output_file):
    """
    验证输出文件的内容
    
    Args:
        output_file (str): 输出文件路径
    """
    logging.info("正在验证输出文件...")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            logging.warning("警告: 输出文件为空")
            return
        
        # 验证前几行的JSON格式
        for i, line in enumerate(lines[:3]):
            try:
                json_obj = json.loads(line.strip())
                if 'file_name' not in json_obj or 'text' not in json_obj:
                    logging.error(f"第 {i+1} 行JSON对象缺少必需字段")
                else:
                    logging.info(f"验证样例 {i+1}: file_name='{json_obj['file_name']}', text='{json_obj['text'][:30]}...'")
            except json.JSONDecodeError as e:
                logging.error(f"第 {i+1} 行JSON格式错误: {str(e)}")
        
        logging.info(f"输出文件验证完成，共 {len(lines)} 行")
        
    except Exception as e:
        logging.error(f"验证输出文件时出错: {str(e)}")

def main():
    """主函数"""
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("CSV to metadata.jsonl Converter for Hugging Face LoRA Training")
    logging.info("=" * 60)
    
    try:
        # 解析命令行参数
        args = parse_args()
        
        logging.info(f"输入参数:")
        logging.info(f"  CSV文件: {args.csv_path}")
        logging.info(f"  图片目录: {args.image_dir}")
        logging.info(f"  输出文件: {args.output_file}")
        logging.info(f"  CSV分隔符: {'TAB' if args.csv_separator == chr(9) else repr(args.csv_separator)}")
        logging.info(f"  图片ID列: {args.image_id_column}")
        logging.info(f"  文本列: {args.text_column}")
        
        # 1. 扫描图片目录
        id_to_path = scan_image_directory(args.image_dir)
        
        # 2. 加载CSV数据
        df = load_csv_data(
            args.csv_path, 
            args.csv_separator, 
            args.image_id_column, 
            args.text_column
        )
        
        # 3. 创建metadata.jsonl文件
        create_metadata_jsonl(
            df, 
            id_to_path, 
            args.output_file,
            args.image_id_column,
            args.text_column
        )
        
        # 4. 验证输出文件
        validate_output_file(args.output_file)
        
        logging.info("转换完成！可以使用以下命令启动Hugging Face LoRA训练:")
        logging.info(f"python train_text_to_image_lora.py --dataset_name=. --caption_column=text \\")
        logging.info(f"  --resolution=512 --train_batch_size=1 --num_train_epochs=100 \\")
        logging.info(f"  --checkpointing_steps=5000 --learning_rate=1e-04 --lr_scheduler=constant \\")
        logging.info(f"  --seed=42 --output_dir=./lora-output")
        
    except KeyboardInterrupt:
        logging.info("用户中断操作")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 