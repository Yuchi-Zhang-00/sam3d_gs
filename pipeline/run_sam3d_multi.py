import os
import argparse
import numpy as np
import torch
from inference import Inference
from help_functions import clean_name, load_image, load_binary_mask, collect_mask_paths

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3D multi-object inference and save outputs to .pt"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="sam3/assets/img.jpg",
        help="Input image path to lift to 3D.",
    )
    parser.add_argument(
        "--mask-root",
        type=str,
        default="sam3/agent_output_multi/masks",
        help="Directory containing mask PNG/JPGs.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="sam-3d-objects/torch_save_pt",
        help="Where to save <parent>_<maskname>.pt files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="hf",
        help="Checkpoint tag, corresponds to ../sam-3d-objects/checkpoints/{tag}/pipeline.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed into Inference.__call__.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Root directory of sam-3d-objects repo. "
            "If not set, will be inferred as <this_script_dir>/../sam-3d-objects."
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.project_root is not None:
        # 如果用户通过命令行显式传入了 --project-root，就直接用它
        project_root = os.path.abspath(args.project_root)
    else:
        # 否则自动推断：假设当前脚本位于 sam3d_gs/pipeline/ 下，
        # sam-3d-objects 位于 sam3d_gs/sam-3d-objects
        project_root = os.path.abspath(os.path.join(script_dir, "..", "sam-3d-objects"))

    print(f"Project root (sam-3d-objects): {project_root}")

    config_path = os.path.join(project_root, "checkpoints", args.tag, "pipeline.yaml")
    print(f"Using config: {config_path}")
    inference = Inference(config_path, compile=False)

    pil_image = load_image(args.image_path)
    image = np.array(pil_image)

    mask_paths = collect_mask_paths(args.mask_root)
    if not mask_paths:
        raise RuntimeError(f"No mask images found under {args.mask_root}")

    # 获取图像文件名（不包含扩展名），用于创建文件夹
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # 清理文件名中的特殊字符
    image_name_clean = clean_name(image_name)
    # 创建图像名对应的文件夹
    image_output_dir = os.path.join(args.save_dir, image_name_clean)

    os.makedirs(image_output_dir, exist_ok=True)

    for i, mask_path in enumerate(mask_paths):
        print(f"[{i+1}/{len(mask_paths)}] running inference on mask: {mask_path}")

        mask = load_binary_mask(mask_path)

        out = inference(image, mask, seed=args.seed)

        # # 构造保存名字：父目录名 + "_" + mask 文件名（无扩展）
        # parent_name_raw = os.path.basename(os.path.dirname(mask_path))
        # parent_name = clean_name(parent_name_raw)
        # mask_stem_raw = os.path.splitext(os.path.basename(mask_path))[0]
        # mask_stem = clean_name(mask_stem_raw)
        # save_name = f"{parent_name}_{mask_stem}.pt"
        # save_path = os.path.join(args.save_dir, save_name)

        # 构造保存名字：使用mask文件名（无扩展名）.pt
        mask_stem_raw = os.path.splitext(os.path.basename(mask_path))[0]
        mask_stem = clean_name(mask_stem_raw)
        save_name = f"{mask_stem}.pt"
        # 保存到图像名对应的文件夹中
        save_path = os.path.join(image_output_dir, save_name)

        torch.save(out, save_path)
        print(f"✅ Saved: {save_path}")

        # 显式释放显存
        del out
        torch.cuda.empty_cache()

    print("✅ All objects processed and saved as .pt")


if __name__ == "__main__":
    main()
