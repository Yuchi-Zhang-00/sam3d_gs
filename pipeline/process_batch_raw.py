import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pipeline.utils import clean_name, load_image, load_binary_mask, collect_mask_paths, compute_fov_from_intrinsics, mesh_rendering, mesh_rendering_with_depth_adjustment
from inference import (
    Inference,
    make_scene,
    transform_mesh,
    ready_gaussian_for_video_rendering,
    render_video,
    render_gs_view,
    interactive_visualizer,
    _fix_gaussian_alignment
)
import copy
from PIL import Image
import cv2
import imageio


def process_single_image(image_path, inference, args):
    """处理单个图片的完整流程"""
    # 确保使用绝对路径
    image_path = os.path.abspath(image_path)
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    
    print(f"Image directory: {image_dir}")
    print(f"Image name: {image_name}")
    
    # 读取图像
    pil_image = load_image(image_path)
    image_bg = np.array(pil_image)
    
    # 构建 mask 目录路径
    mask_dir = os.path.join(image_dir, 'masks')
    
    # 从 mask 目录收集 mask 路径
    mask_paths = collect_mask_paths(mask_dir)

    
    # 构建输出目录
    assets_dir = os.path.join(image_dir, '3d_assets')
    pt_dir = os.path.join(image_dir, 'pt')
    gif_dir = os.path.join(image_dir, 'gif')
    
    if not mask_paths:
        print(f"Warning: No mask images found in {mask_dir}")
        print("Creating placeholder files and continuing...")
        os.makedirs(assets_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        return
    
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    
   
    
    mask_names = []
    masks = []
    original_sizes = []
    outputs = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 处理每个mask
    for i, mask_path in enumerate(mask_paths):
        print(f"\n[{i+1}/{len(mask_paths)}] Processing mask: {mask_path}")
        
        # 加载掩码
        mask_ = np.array(Image.open(mask_path).convert("L"))
        mask = np.where(mask_ > 0, 1, 0).astype("uint8")
        
        mask_unique_values = np.unique(mask)
        print(f"Mask unique values: {mask_unique_values}")
        
        size_ori = np.sum(mask)
        original_sizes.append(size_ori)
        masks.append(mask)
        
        
        
        # 构造保存名字
        mask_stem_raw = os.path.splitext(os.path.basename(mask_path))[0]
        mask_stem = clean_name(mask_stem_raw)
        mask_names.append(mask_stem)
        
        # 保存二值化掩码
        cv2.imwrite(os.path.join(image_dir, f"{mask_stem}_binary.png"), mask * 255)
        
        # 保存路径
        save_name = f"{mask_stem}.pt"
        save_path = os.path.join(pt_dir, save_name)
        
        # 检查是否已存在处理结果
        if os.path.exists(save_path):
            print(f"Loading existing .pt file: {save_path}")
            out = torch.load(save_path, map_location=device, weights_only=False)
        else:
            print(f"Running inference on mask...")
            out = inference(image_bg, mask, seed=args.seed)
            torch.save(out, save_path)
            print(f"Saved inference result: {save_path}")
        
        # 输出out 的dict键
        print(f"  Output keys: {list(out.keys())}")
        
        outputs.append(out)

        # 只用 make_scene，不做 ready_gaussian_for_video_rendering
        single_scene = make_scene(out)

        # stem = os.path.splitext(os.path.basename(p))[0]
        # single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
        # single_scene.save_ply(single_ply_path)
        # print(f"🟢 Saved single-object PLY: {single_ply_path}")

        # if args.export_gif:
        #     video = render_video(
        #         single_scene,
        #         r=1,
        #         fov=60,
        #         resolution=512,
        #     )["color"]

        #     single_gif_path = os.path.join(gif_dir, f"{mask_stem}.gif")
        #     imageio.mimsave(
        #         single_gif_path,
        #         video,
        #         format="GIF",
        #         duration=1000 / 30,  # 30fps
        #         loop=0,
        #     )
        #     print(f"🎞️ Saved single-object GIF: {single_gif_path}")

        # 如果显存很紧张，可以在这里 del single_scene / video 等
        del single_scene

    print("✅ All single-object scenes exported.")

    # =========================
    # 2️⃣ 合并多对象场景：PLY (+ 可选 GIF)
    # =========================
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    # gauss_dir = os.path.join(project_root, "gaussians", "multi")
    # os.makedirs(gauss_dir, exist_ok=True)

    ply_path = os.path.join(assets_dir, f"{image_name}.ply")
    scene_gs.save_ply(ply_path)
    print(f"✅ Saved merged PLY: {ply_path}")

    # if args.export_gif:
    video = render_video(
        scene_gs,
        r=2,
        fov=60,
        resolution=1024,
    )["color"]

    gif_path = os.path.join(gif_dir, f"{image_name}.gif")
    imageio.mimsave(
        gif_path,
        video,
        format="GIF",
        duration=1000 / 30,  # 30fps
        loop=0,
    )
    print(f"✅ Saved merged GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3D multi-object inference and save outputs to .pt, and reconstruct single object Gaussian .ply"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="Sam-3d-objects",
        help="Root directory of sam-3d-objects project.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="sam-3d-objects/torch_save_pt",
        help="Directory containing *.pt files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing image folders.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="input_image.png",
        help="Name of the image file to process in each folder.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="hf",
        help="Checkpoint tag, corresponds to ../Sam-3d-objects/checkpoints/{tag}/pipeline.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed into Inference.__call__.",
    )
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(args.input_dir):
        input_dir = os.path.dirname(os.path.abspath(args.input_dir))
    else:
        input_dir = os.path.abspath(args.input_dir)

    
    if args.project_root is not None:
        project_root = os.path.abspath(args.project_root)
    else:
        project_root = os.path.abspath(os.path.join(script_dir, "..", "Sam-3d-objects"))

    print(f"Project root (Sam-3d-objects): {project_root}")
    print(f"Input directory: {input_dir}")
    print(f"Looking for image files named: {args.image_name}")

    # 加载模型（只加载一次）
    config_path = os.path.join(project_root, "checkpoints", args.tag, "pipeline.yaml")
    print(f"Loading model from config: {config_path}")
    inference = Inference(config_path, compile=False)
    print("Model loaded successfully")
    
    # # 查找所有input_image.png文件
    # # 从 masks.txt 读取路径
    # # 假设 masks.txt 每一行格式为: /path/to/traj0_x/masks 数量
    # masks_info_file = "/home/discover/sam3d_gs/masks.txt"
    # target_filename = args.image_name # 假设是 "input_image.png"
    # image_files = []

    # if os.path.exists(masks_info_file):
    #     with open(masks_info_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
                
    #             # 拆分路径和数量（取第一部分为路径）
    #             mask_path = line.rsplit(' ', 1)[0]
                
    #             # 拼接路径逻辑：
    #             # 方法 A: 假设 input_image.png 在 masks 文件夹的同级目录
    #             # 即 /.../traj0_13/masks -> /.../traj0_13/input_image.png
    #             parent_dir = os.path.dirname(mask_path)
    #             image_path = os.path.join(parent_dir, target_filename)
                
    #             # 检查文件是否真的存在，防止 txt 记录过时
    #             if os.path.exists(image_path):
    #                 image_files.append(image_path)
    #                 print(f"Added: {image_path}") # 调试用
    #             else:
    #                 print(f"Warning: File not found: {image_path}")
    # else:
    #     print(f"Error: {masks_info_file} not found. Please run the scan script first.")

    # print(f"Found {len(image_files)} image files based on masks.txt")

    # 查找所有input_image.png文件
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == args.image_name:
                image_path = os.path.join(root, file)
                print(f"Found image file: {image_path}")
                image_files.append(image_path)
    
    print(f"Found {len(image_files)} image files to process")
    
    if len(image_files) == 0:
        print(f"No {args.image_name} files found in {input_dir}")
        print("Directory structure:")
        for root, dirs, files in os.walk(input_dir):
            level = root.replace(input_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f'{subindent}{file}')
        return
    
    # 处理每个图像文件
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing image {idx}/{len(image_files)}")
        print(f"Image path: {image_path}")
        print(f"{'='*80}")
        
        try:
            process_single_image(image_path, inference, args)
            print(f"✓ Successfully processed: {image_path}")
        except Exception as e:
            print(f"✗ Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"All processing completed! Processed {len(image_files)} images")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
