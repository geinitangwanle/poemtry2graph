# 批量图片处理脚本

这是一个用于批量处理图片的Python脚本，可以将任意尺寸的图片缩放并裁剪成512x512像素的方形图片。

## 功能特性

- ✅ 支持多种图片格式：JPG、JPEG、PNG、BMP、TIFF、WebP
- ✅ 智能缩放：将图片的最短边缩放到512像素，保持原始宽高比
- ✅ 中心裁剪：从图片中心裁剪出512x512像素的方形图片
- ✅ 批量处理：一次性处理整个文件夹中的所有图片
- ✅ 自动创建输出文件夹
- ✅ 详细的处理日志和进度显示
- ✅ 错误处理和统计报告

## 安装依赖

首先确保安装了所需的依赖包：

```bash
pip install -r requirements.txt
```

或者直接安装Pillow：

```bash
pip install Pillow>=10.0.0
```

## 使用方法

### 基本用法

```bash
python batch_image_processor.py --input_folder /path/to/input/images --output_folder /path/to/output/images
```
python batch_image_processor.py --input_folder /Users/tangren/Documents/poemtry2graph/Paint4Poem-Web-famous-subset/images --output_folder /Users/tangren/Documents/poemtry2graph/Paint4Poem-Web-famous-subset/processed_image 
### 参数说明

- `--input_folder`：必需参数，原始图片所在的文件夹路径
- `--output_folder`：必需参数，处理后图片要保存的文件夹路径
- `--target_size`：可选参数，目标图片尺寸（默认512像素）

### 使用示例

```bash
# 基本使用
python batch_image_processor.py --input_folder ./raw_images --output_folder ./processed_images

# 指定不同的目标尺寸
python batch_image_processor.py --input_folder ./raw_images --output_folder ./processed_images --target_size 256

# 使用绝对路径
python batch_image_processor.py --input_folder /Users/username/Pictures/raw --output_folder /Users/username/Pictures/processed
```

## 处理流程

1. **验证输入**：检查输入文件夹是否存在
2. **创建输出文件夹**：如果输出文件夹不存在，自动创建
3. **扫描图片文件**：查找输入文件夹中所有支持的图片格式
4. **批量处理**：对每张图片执行以下操作：
   - 将图片模式转换为RGB（处理透明通道等问题）
   - 计算缩放比例，使最短边达到目标尺寸
   - 缩放图片，保持原始宽高比
   - 从中心裁剪出方形图片
   - 以JPEG格式保存（质量95%）
5. **结果统计**：显示处理成功和失败的图片数量

## 支持的图片格式

- `.jpg` / `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.webp`

## 输出说明

- 所有处理后的图片都会以JPEG格式保存（.jpg扩展名）
- 保持原有的文件名（不包括扩展名）
- 图片质量设置为95%，确保高质量输出
- 所有输出图片都是512x512像素的方形图片

## 日志信息

脚本会显示详细的处理信息：

```
2024-01-01 12:00:00,000 - INFO - 创建输出文件夹: ./processed_images
2024-01-01 12:00:01,000 - INFO - 找到 10 张图片需要处理
2024-01-01 12:00:01,100 - INFO - 正在处理 [1/10]: image1.jpg
2024-01-01 12:00:01,200 - INFO - 成功处理: image1.jpg -> image1.jpg
...
2024-01-01 12:00:10,000 - INFO - 处理完成！成功: 10 张，失败: 0 张
```

## 注意事项

1. 确保有足够的磁盘空间来存储处理后的图片
2. 处理大量图片时可能需要一些时间，请耐心等待
3. 如果输入文件夹中没有找到支持的图片文件，脚本会显示警告信息
4. 处理过程中如果遇到损坏的图片文件，会跳过并记录错误日志

## 故障排除

### 常见错误

1. **"输入文件夹不存在"**：请检查输入文件夹路径是否正确
2. **"未找到支持的图片文件"**：请确认输入文件夹中包含支持格式的图片文件
3. **权限错误**：请确保对输入和输出文件夹有相应的读写权限

### 性能优化

- 对于大量图片的批量处理，建议使用SSD硬盘以提高I/O性能
- 可以考虑修改脚本以支持多线程处理（需要额外开发） 