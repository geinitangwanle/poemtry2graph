#!/usr/bin/env python3
"""
测试批量图片处理脚本的简单示例
"""

import os
import sys
from pathlib import Path
from PIL import Image
import random

def create_test_images(test_folder, num_images=3):
    """
    创建一些测试图片
    
    Args:
        test_folder (Path): 测试文件夹路径
        num_images (int): 要创建的图片数量
    """
    # 创建测试文件夹
    test_folder.mkdir(exist_ok=True)
    
    # 创建不同尺寸的测试图片
    test_configs = [
        (800, 600, 'landscape.jpg'),    # 横向图片
        (600, 800, 'portrait.png'),     # 纵向图片
        (1024, 1024, 'square.jpeg'),   # 方形图片
    ]
    
    for width, height, filename in test_configs[:num_images]:
        # 创建随机颜色的图片
        color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
        
        image = Image.new('RGB', (width, height), color)
        
        # 添加一些简单的图案（画十字线）
        pixels = image.load()
        for x in range(width):
            pixels[x, height//2] = (255, 255, 255)  # 水平线
        for y in range(height):
            pixels[width//2, y] = (255, 255, 255)   # 垂直线
        
        # 保存图片
        image_path = test_folder / filename
        image.save(image_path)
        print(f"创建测试图片: {image_path} ({width}x{height})")

def main():
    """
    主测试函数
    """
    print("=" * 50)
    print("批量图片处理脚本测试")
    print("=" * 50)
    
    # 设置测试路径
    test_input_folder = Path("test_images_input")
    test_output_folder = Path("test_images_output")
    
    # 创建测试图片
    print("\n1. 创建测试图片...")
    create_test_images(test_input_folder)
    
    # 显示使用说明
    print(f"\n2. 测试图片已创建在: {test_input_folder.absolute()}")
    print(f"   现在可以运行以下命令来测试脚本：")
    print(f"\n   python batch_image_processor.py \\")
    print(f"       --input_folder {test_input_folder} \\")
    print(f"       --output_folder {test_output_folder}")
    
    print(f"\n3. 处理完成后，检查输出文件夹: {test_output_folder.absolute()}")
    print(f"   应该包含3张512x512像素的处理后图片")
    
    print("\n" + "=" * 50)
    print("测试准备完成！")
    print("=" * 50)

if __name__ == "__main__":
    main() 