import os
import glob
import argparse

import torch

from inference import (
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
    interactive_visualizer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Load saved *.pt and reconstruct single & multi-object Gaussian .ply"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="sam-3d-objects",
        help="Root directory of sam-3d-objects project.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="sam-3d-objects/torch_save_pt",
        help="Directory containing *.pt files.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="sam3/assets/img.jpg",
        help="Original image path (used only to derive IMAGE_NAME).",
    )
    parser.add_argument(
        "--export-gif",
        action="store_true",
        help="If set, render GIFs for each object and the merged scene.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    image_path = args.image_path
    image_name = os.path.basename(os.path.dirname(image_path))

    # 这里不再限定 object_*.pt，而是把 save-dir 下所有 .pt 都吃掉
    paths = sorted(glob.glob(os.path.join(args.save_dir, "*.pt")))
    if not paths:
        raise RuntimeError(f"No .pt found under {args.save_dir}")

    print(f"Found {len(paths)} .pt files:")
    for p in paths:
        print("  ", p)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 单物体输出目录
    single_gauss_dir = os.path.join(project_root, "gaussians", "single")
    os.makedirs(single_gauss_dir, exist_ok=True)

    # 合并场景要用到的 outputs
    outputs = []

    if args.export_gif:
        import imageio

    # =========================
    # 1️⃣ 遍历每个 .pt：导出单物体 PLY (+ 可选 GIF)
    # =========================
    for idx, p in enumerate(paths):
        print(f"[{idx+1}/{len(paths)}] loading {p}")
        out = torch.load(p, map_location=device)
        # 输出out 的dict键
        print(f"  Output keys: {list(out.keys())}")
        
        outputs.append(out)

        # 只用 make_scene，不做 ready_gaussian_for_video_rendering
        single_scene = make_scene(out)

        stem = os.path.splitext(os.path.basename(p))[0]
        single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
        single_scene.save_ply(single_ply_path)
        print(f"🟢 Saved single-object PLY: {single_ply_path}")

        if args.export_gif:
            video = render_video(
                single_scene,
                r=1,
                fov=60,
                resolution=512,
            )["color"]

            single_gif_path = os.path.join(single_gauss_dir, f"{stem}.gif")
            imageio.mimsave(
                single_gif_path,
                video,
                format="GIF",
                duration=1000 / 30,  # 30fps
                loop=0,
            )
            print(f"🎞️ Saved single-object GIF: {single_gif_path}")

        # 如果显存很紧张，可以在这里 del single_scene / video 等
        del single_scene

    print("✅ All single-object scenes exported.")

    # =========================
    # 2️⃣ 合并多对象场景：PLY (+ 可选 GIF)
    # =========================
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    gauss_dir = os.path.join(project_root, "gaussians", "multi")
    os.makedirs(gauss_dir, exist_ok=True)

    ply_path = os.path.join(gauss_dir, f"{image_name}.ply")
    scene_gs.save_ply(ply_path)
    print(f"✅ Saved merged PLY: {ply_path}")

    if args.export_gif:
        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            resolution=512,
        )["color"]

        gif_path = os.path.join(gauss_dir, f"{image_name}.gif")
        imageio.mimsave(
            gif_path,
            video,
            format="GIF",
            duration=1000 / 30,  # 30fps
            loop=0,
        )
        print(f"✅ Saved merged GIF: {gif_path}")


if __name__ == "__main__":
    main()
