import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

def merge_mask_with_image(
    image_path: str,
    masks_folder: str,
    threshold: int = 127,
    background_color: tuple = (0, 0, 0),
    resize_mode: str = 'match_image'
):
    """
    将黑白 mask 与原始图片合成，直接覆盖原 mask 文件
    """
    # 加载原始图片
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # 获取所有 mask 文件
    mask_files = [f for f in os.listdir(masks_folder) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f'📁 原始图片：{image_path} {img_array.shape}')
    print(f'📂 找到 {len(mask_files)} 个 mask 文件')
    print(f'🔲 阈值：{threshold} | 背景色：{background_color}')
    print('-' * 50)
    
    for mask_file in mask_files:
        mask_path = os.path.join(masks_folder, mask_file)
        
        # 加载 mask
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        # 尺寸匹配
        if resize_mode == 'match_image':
            if mask_array.shape != img_array.shape[:2]:
                mask = mask.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
                mask_array = np.array(mask)
            target_shape = img_array.shape[:2]
        else:  # match_mask
            if img_array.shape[:2] != mask_array.shape:
                img = img.resize((mask_array.shape[1], mask_array.shape[0]), Image.BILINEAR)
                img_array = np.array(img)
            target_shape = mask_array.shape
        
        # 创建输出图片（初始为背景色）
        output = np.full_like(img_array, background_color)
        
        # mask 白色部分显示原图像素
        white_mask = mask_array > threshold
        output[white_mask] = img_array[white_mask]
        
        # 直接覆盖原 mask 文件
        Image.fromarray(output).save(mask_path)
        
        # 统计信息
        white_ratio = white_mask.sum() / white_mask.size * 100
        print(f'✓ {mask_file:25s} | 白色占比：{white_ratio:6.2f}% | {target_shape}')
    
    print('-' * 50)
    print(f'✅ 完成！已覆盖原 mask 文件')


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将 masks 文件夹下的黑白 mask 与 input_image.png 合成，直接覆盖原 mask'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='输入目录路径（包含 input_image.png 和 masks 文件夹）'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=127,
        help='黑白阈值 (默认：127)'
    )
    parser.add_argument(
        '--background',
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=('R', 'G', 'B'),
        help='背景颜色 RGB 值 (默认：0 0 0 黑色)'
    )
    
    args = parser.parse_args()
    
    # 构建路径
    base_path = args.input_dir
    image_path = os.path.join(base_path, 'input_image.png')
    masks_folder = os.path.join(base_path, 'masks')
    
    # 验证路径
    if not os.path.exists(image_path):
        print(f'❌ 错误：找不到图片 {image_path}')
        exit(1)
    
    if not os.path.exists(masks_folder):
        print(f'❌ 错误：找不到 masks 文件夹 {masks_folder}')
        exit(1)
    
    # 执行合成
    merge_mask_with_image(
        image_path=image_path,
        masks_folder=masks_folder,
        threshold=args.threshold,
        background_color=tuple(args.background),
        resize_mode='match_image'
    )