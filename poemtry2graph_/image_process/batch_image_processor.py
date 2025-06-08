#!/usr/bin/env python3
"""
批量图片处理脚本
功能：将图片缩放并裁剪成512x512像素的方形图片

使用方法：
python batch_image_processor.py --input_folder /path/to/input --output_folder /path/to/output
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def setup_output_folder(output_path):
    """
    检查并创建输出文件夹
    
    Args:
        output_path (Path): 输出文件夹路径
    """
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建输出文件夹: {output_path}")
    else:
        logger.info(f"输出文件夹已存在: {output_path}")

def resize_and_crop_image(image, target_size=512):
    """
    将图片的最短边缩放到目标尺寸，然后从中心裁剪成正方形
    
    Args:
        image (PIL.Image): 输入图片对象
        target_size (int): 目标尺寸（像素）
    
    Returns:
        PIL.Image: 处理后的图片对象
    """
    # 获取原始图片尺寸
    original_width, original_height = image.size
    logger.debug(f"原始尺寸: {original_width}x{original_height}")
    
    # 计算缩放比例，以最短边为准
    min_dimension = min(original_width, original_height)
    scale_ratio = target_size / min_dimension
    
    # 计算新的尺寸
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    logger.debug(f"缩放后尺寸: {new_width}x{new_height}")
    
    # 缩放图片，保持宽高比
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 计算裁剪区域，从中心裁剪
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    logger.debug(f"裁剪区域: ({left}, {top}, {right}, {bottom})")
    
    # 裁剪图片
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image

def process_single_image(input_path, output_path, target_size=512):
    """
    处理单张图片
    
    Args:
        input_path (Path): 输入图片路径
        output_path (Path): 输出图片路径
        target_size (int): 目标尺寸
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 打开图片
        with Image.open(input_path) as image:
            # 确保图片是RGB模式（处理透明通道等问题）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 处理图片
            processed_image = resize_and_crop_image(image, target_size)
            
            # 保存处理后的图片
            processed_image.save(output_path, 'JPEG', quality=95)
            
        logger.info(f"成功处理: {input_path.name} -> {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"处理图片 {input_path.name} 时出错: {str(e)}")
        return False

def get_image_files(input_folder):
    """
    获取输入文件夹中所有支持的图片文件
    
    Args:
        input_folder (Path): 输入文件夹路径
    
    Returns:
        list: 图片文件路径列表
    """
    image_files = []
    
    for file_path in input_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file_path)
    
    return sorted(image_files)

def main():
    """
    主函数
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量处理图片：缩放并裁剪成512x512像素的方形图片')
    parser.add_argument('--input_folder', type=str, required=True, 
                       help='输入图片文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='输出图片文件夹路径')
    parser.add_argument('--target_size', type=int, default=512, 
                       help='目标图片尺寸（默认512像素）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 转换为Path对象
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    target_size = args.target_size
    
    # 验证输入文件夹是否存在
    if not input_folder.exists():
        logger.error(f"输入文件夹不存在: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"输入路径不是文件夹: {input_folder}")
        sys.exit(1)
    
    # 设置输出文件夹
    setup_output_folder(output_folder)
    
    # 获取所有图片文件
    image_files = get_image_files(input_folder)
    
    if not image_files:
        logger.warning(f"在输入文件夹中未找到支持的图片文件")
        logger.info(f"支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    logger.info(f"找到 {len(image_files)} 张图片需要处理")
    
    # 批量处理图片
    success_count = 0
    failed_count = 0
    
    for i, input_path in enumerate(image_files, 1):
        logger.info(f"正在处理 [{i}/{len(image_files)}]: {input_path.name}")
        
        # 构建输出文件路径，保持原有文件名但统一为.jpg格式
        output_filename = input_path.stem + '.jpg'
        output_path = output_folder / output_filename
        
        # 处理图片
        if process_single_image(input_path, output_path, target_size):
            success_count += 1
        else:
            failed_count += 1
    
    # 输出处理结果统计
    logger.info(f"处理完成！成功: {success_count} 张，失败: {failed_count} 张")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main() 