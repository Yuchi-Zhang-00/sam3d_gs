import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm  # 进度条支持

def apply_colormap_to_depth(depth_gray, colormap=cv2.COLORMAP_JET):
    """
    将灰度深度图应用色彩映射
    
    Args:
        depth_gray: 灰度深度图 (0-255)
        colormap: OpenCV色彩映射，可选值：
            cv2.COLORMAP_JET (默认)
            cv2.COLORMAP_HOT
            cv2.COLORMAP_VIRIDIS
            cv2.COLORMAP_PLASMA
            cv2.COLORMAP_TURBO
            cv2.COLORMAP_RAINBOW
            cv2.COLORMAP_OCEAN
            cv2.COLORMAP_SPRING
            cv2.COLORMAP_SUMMER
            cv2.COLORMAP_AUTUMN
            cv2.COLORMAP_WINTER
            cv2.COLORMAP_BONE
            cv2.COLORMAP_COOL
            cv2.COLORMAP_HSV
            cv2.COLORMAP_PINK
            cv2.COLORMAP_CIVIDIS
            cv2.COLORMAP_TWILIGHT
            cv2.COLORMAP_TWILIGHT_SHIFTED
            cv2.COLORMAP_INFERNO
            cv2.COLORMAP_MAGMA
    Returns:
        彩色深度图 (BGR格式)
    """
    # 确保是8位图像
    if depth_gray.dtype != np.uint8:
        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = depth_normalized.astype(np.uint8)
    else:
        depth_8bit = depth_gray
    
    # 应用色彩映射
    depth_colored = cv2.applyColorMap(depth_8bit, colormap)
    
    return depth_colored

def process_depth_images(root_path, output_suffix="_colored", colormap_name="JET"):
    """
    递归处理深度图并保存彩色版本
    
    Args:
        root_path: 根目录路径
        output_suffix: 输出文件名的后缀
        colormap_name: 色彩映射名称
    """
    root_path = Path(root_path).resolve()
    
    if not root_path.exists():
        print(f"错误：路径不存在 - {root_path}")
        return
    
    if not root_path.is_dir():
        print(f"错误：路径不是目录 - {root_path}")
        return
    
    # 色彩映射字典
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "TURBO": cv2.COLORMAP_TURBO,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SPRING": cv2.COLORMAP_SPRING,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "WINTER": cv2.COLORMAP_WINTER,
        "BONE": cv2.COLORMAP_BONE,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "CIVIDIS": cv2.COLORMAP_CIVIDIS,
        "TWILIGHT": cv2.COLORMAP_TWILIGHT,
        "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
    }
    
    selected_colormap = colormap_dict.get(colormap_name.upper(), cv2.COLORMAP_JET)
    
    # 查找所有depth_ori_visual.png文件
    depth_files = list(root_path.rglob("depth_ori_visual.png"))
    
    if not depth_files:
        print(f"未找到任何 depth_ori_visual.png 文件在 {root_path}")
        return
    
    print(f"找到 {len(depth_files)} 个 depth_ori_visual.png 文件")
    print(f"使用色彩映射: {colormap_name}")
    print("-" * 50)
    
    processed_count = 0
    failed_count = 0
    
    # 处理每个文件（使用进度条）
    for depth_file in tqdm(depth_files, desc="处理深度图"):
        try:
            # 读取灰度深度图
            depth_gray = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            
            if depth_gray is None:
                print(f"警告：无法读取 {depth_file.relative_to(root_path)}")
                failed_count += 1
                continue
            
            # 应用色彩映射
            depth_colored = apply_colormap_to_depth(depth_gray, selected_colormap)
            
            # 构建输出路径
            output_name = f"{depth_file.stem}{output_suffix}{depth_file.suffix}"
            output_path = depth_file.parent / output_name
            
            # 保存彩色深度图
            success = cv2.imwrite(str(output_path), depth_colored)
            
            if success:
                processed_count += 1
            else:
                print(f"警告：无法保存 {output_path.relative_to(root_path)}")
                failed_count += 1
                
        except Exception as e:
            print(f"处理 {depth_file.relative_to(root_path)} 时出错: {e}")
            failed_count += 1
    
    print("-" * 50)
    print(f"完成! 成功处理: {processed_count} 个文件, 失败: {failed_count} 个文件")
    print(f"彩色深度图保存在原文件同目录，后缀为 '{output_suffix}'")

def visualize_sample(root_path, colormap_name="JET"):
    """
    预览一个深度图转换效果
    """
    import matplotlib.pyplot as plt
    
    root_path = Path(root_path).resolve()
    depth_files = list(root_path.rglob("depth_ori_visual.png"))
    
    if not depth_files:
        print("没有找到文件用于预览")
        return
    
    # 取第一个文件预览
    sample_file = depth_files[0]
    print(f"预览文件: {sample_file.relative_to(root_path)}")
    
    # 读取并转换
    depth_gray = cv2.imread(str(sample_file), cv2.IMREAD_GRAYSCALE)
    
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "TURBO": cv2.COLORMAP_TURBO,
    }
    selected_colormap = colormap_dict.get(colormap_name.upper(), cv2.COLORMAP_JET)
    
    depth_colored = apply_colormap_to_depth(depth_gray, selected_colormap)
    
    # 显示对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(depth_gray, cmap='gray')
    axes[0].set_title('原始灰度深度图')
    axes[0].axis('off')
    
    axes[1].imshow(depth_colored[:,:,::-1])  # BGR转RGB
    axes[1].set_title(f'彩色深度图 ({colormap_name})')
    axes[1].axis('off')
    
    # 显示颜色条
    axes[2].imshow(depth_colored[:,:,::-1])
    axes[2].set_title('带深度值参考')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将灰度深度图转换为彩色深度图")
    parser.add_argument("path", help="要搜索的根目录路径", default="/home/discover/sam3d_gs/data", nargs="?")
    parser.add_argument("--suffix", help="输出文件后缀", default="_colored")
    parser.add_argument("--colormap", help="色彩映射类型", default="JET", 
                       choices=["JET", "HOT", "VIRIDIS", "PLASMA", "TURBO", "RAINBOW", 
                               "OCEAN", "SPRING", "SUMMER", "AUTUMN", "WINTER", 
                               "BONE", "COOL", "HSV", "PINK", "CIVIDIS", 
                               "TWILIGHT", "TWILIGHT_SHIFTED", "INFERNO", "MAGMA"])
    parser.add_argument("--preview", action="store_true", help="预览转换效果（显示第一个文件）")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要进行的更改，不实际处理")
    
    args = parser.parse_args()
    
    # 如果需要安装tqdm，可以取消注释下面这行
    # print("提示: 如需进度条支持，请运行: pip install tqdm")
    
    if args.preview:
        # 预览模式
        try:
            import matplotlib.pyplot as plt
            visualize_sample(args.path, args.colormap)
        except ImportError:
            print("需要matplotlib来预览，请安装: pip install matplotlib")
    elif args.dry_run:
        # 模拟运行模式
        root_path = Path(args.path).resolve()
        depth_files = list(root_path.rglob("depth_ori_visual.png"))
        print(f"[DRY-RUN] 将在 {root_path} 中找到 {len(depth_files)} 个文件")
        print(f"[DRY-RUN] 输出文件名示例: depth_ori_visual.png -> depth_ori_visual{args.suffix}.png")
        print(f"[DRY-RUN] 色彩映射: {args.colormap}")
    else:
        # 实际处理
        process_depth_images(args.path, args.suffix, args.colormap)