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
    interactive_visualizer,
    _fix_gaussian_alignment
)
import copy
from PIL import Image
import cv2
import imageio
        
def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3D multi-object inference and save outputs to .pt, and reconstruct single object Gaussian .ply"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="Sam-3d-objects",
        help="Root directory of Sam-3d-objects project.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="Sam-3d-objects/torch_save_pt",
        help="Directory containing *.pt files.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="Sam3/assets/img.jpg",
        help="Original image path (Input image path to lift to 3D.).",
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

    if args.project_root is not None:
        # 如果用户通过命令行显式传入了 --project-root，就直接用它
        project_root = os.path.abspath(args.project_root)
    else:
        # 否则自动推断：假设当前脚本位于 Sam3d_gs/pipeline/ 下，
        # Sam-3d-objects 位于 Sam3d_gs/Sam-3d-objects
        project_root = os.path.abspath(os.path.join(script_dir, "..", "Sam-3d-objects"))

    print(f"Project root (Sam-3d-objects): {project_root}")

    config_path = os.path.join(project_root, "checkpoints", args.tag, "pipeline.yaml")
    print(f"Using config: {config_path}")
    inference = Inference(config_path, compile=False)

    # 读取图像
    pil_image = load_image(args.image_path)
    image_bg = np.array(pil_image)

    # mask_paths = collect_mask_paths(args.mask_root)
    # 获取图像文件所在目录
    image_dir = os.path.dirname(args.image_path)
    # 构建 mask 目录路径（与 image_path 同级）
    mask_dir = os.path.join(image_dir, 'masks')
    # 从 mask 目录收集 mask 路径
    mask_paths = collect_mask_paths(mask_dir)
    # 构建 3D 资产 输出目录路径（与 image_path 同级）
    assets_dir = os.path.join(image_dir, '3d_assets')
    # 构建 pt 保存路径（与 image_path 同级）
    pt_dir = os.path.join(image_dir, 'pt')

    if not mask_paths:
        # raise RuntimeError(f"No mask images found under {args.mask_root}")
        raise RuntimeError(f"No mask images found under {mask_dir}")

    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    extrinsics = np.load(os.path.join(image_dir, 'extrinsic.npy'))
    intrinsics = np.load(os.path.join(image_dir, 'intrinsic.npy'))
    depth_anysplat      = np.load(os.path.join(image_dir, 'depth.npy'))
    scale_factor      = np.load(os.path.join(image_dir, 'scale.npy')) if os.path.exists(os.path.join(image_dir, 'scale.npy')) else None

    # 打印信息 (已修正标签错误)
    print("extrinsics", type(extrinsics), "\n", extrinsics) 
    print("intrinsics", type(intrinsics), "\n", intrinsics) # 修正了这里的字符串
    print("depth_anysplat shape", depth_anysplat.shape)
    print(f"anyplat mean_depth_ori: {depth_anysplat.mean():.4f}, min_depth_ori: {depth_anysplat.min():.4f}, max_depth_ori: {depth_anysplat.max():.4f}")

    image_size = (448, 448)

    fx_pixels = intrinsics[0, 0]
    fy_pixels = intrinsics[1, 1]
    print(f"像素焦距: fx={fx_pixels:.2f}, fy={fy_pixels:.2f}")

    # 3. 使用像素焦距计算真实的FOV
    fov_x, fov_y = compute_fov_from_intrinsics(fx_pixels, fy_pixels, image_size, degrees=True)
    print(f"垂直FOV: {fov_y:.2f}度")
    print(f"水平FOV: {fov_x:.2f}度")

    mask_names = []
    masks = []
    original_sizes = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, mask_path in enumerate(mask_paths):
        print(f"[{i+1}/{len(mask_paths)}] running inference on mask: {mask_path}")

        # mask = load_binary_mask(mask_path)
        mask_ = np.array(Image.open(mask_path).convert("L"))
        mask = np.where(mask_ > 0, 1, 0).astype("uint8")
        print(f"mask 的value {np.unique(mask)}")
        size_ori = np.sum(mask)
        original_sizes.append(size_ori)
        masks.append(mask)
        print(f"depth_anysplat shape: {depth_anysplat.shape}")
        print(f"depth_anysplat dtype: {depth_anysplat.dtype}")
        depth_fg = depth_anysplat[mask]
        mean_depth_ori = depth_fg.mean()
        min_depth_ori = depth_fg.min()
        max_depth_ori = depth_fg.max()
        print(f"anyplat mean_depth_ori: {mean_depth_ori:.4f}, min_depth_ori: {min_depth_ori:.4f}, max_depth_ori: {max_depth_ori:.4f}")
        depth_normalized = ((depth_fg - depth_fg.min()) / (depth_fg.max() - depth_fg.min()) * 255).astype(np.uint8)
        # imageio.imwrite( './depth_visual_1.png', depth_normalized)
        # 构造保存名字：使用mask文件名（无扩展名）.pt
        mask_stem_raw = os.path.splitext(os.path.basename(mask_path))[0]
        mask_stem = clean_name(mask_stem_raw)
        mask_names.append(mask_stem)
        cv2.imwrite(os.path.join(image_dir, f"{mask_stem}_binary.png"),mask*255)
        save_name = f"{mask_stem}.pt"
        # 保存到图像名对应的文件夹中
        save_path = os.path.join(pt_dir, save_name)

        if os.path.exists(save_path):
            print(f"✅ Loading existing .pt file: {save_path}")
            out = torch.load(save_path, map_location=device,weights_only=False)
        else:
            out = inference(image_bg, mask, seed=args.seed)       
            torch.save(out, save_path)
            print(f"✅ Saved: {save_path}")

        gs_origin = copy.deepcopy(out["gs"])
        gs_origin.save_ply(os.path.join(assets_dir, f"{mask_stem}_gs_origin.ply"))

        # single_scene = make_scene(out)
        # # 改动x，z轴朝向与anysplat的结果对齐 （x右y下z前）
        # xyz = single_scene.get_xyz
        # xyz_cv = xyz.clone()
        # xyz_cv[:, 1] = -xyz[:, 1]  # Y轴翻转      
        # xyz_cv[:, 0] = -xyz[:, 0]  # X轴翻转
        # single_scene.from_xyz(xyz_cv)

        # stem = os.path.splitext(os.path.basename(p))[0]
        # single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
        single_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_sam3d_target.ply")
        # single_scene.save_ply(single_ply_path)
        # print(f"🟢 Saved single-object PLY: {single_ply_path}")
        
        # 输出未经transform的Gaussian
        untransforrmed_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_untransformed.ply")
        # out["gs"].save_ply(untransforrmed_ply_path)
        # print(f"🟢 Saved untransformed single-object PLY: {untransforrmed_ply_path}")
        
        # 打印object的pose，scale
        rotation_output = out['rotation'].cpu().numpy()
        translation_output = out['translation'].cpu().numpy()
        scale_output = out['scale'].squeeze(0).cpu().numpy()
        scale_anysplat = scale_output[0]
        print(f"anysplat scale of {mask_stem} : {scale_anysplat}")
        print(f"total scale of {mask_stem} : {scale_anysplat*0.33980582524271846}")
        # print(f"type scale_output: {type(scale_output)}, scale_output: {scale_output}")
        # print(f" rotation: {out['rotation']}")
        # print(f" translation: {out['translation']}")
        # print(f" scale: {out['scale']}")            

        # setting_scale = 0.246/0.722
        setting_scale = 1
        
        if out['glb']:              # 如果输出包含mesh
            mesh = out['glb']
            untransformed_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_untransformed.obj")
            # mesh.export(untransformed_mesh_path)
            print(f"🟢 Saved untransformed object Mesh: {untransformed_mesh_path}")
            
            # 输出的mesh在trimesh坐标系下，将其转向输出Gaussian的初始坐标系 
            rot_coordinate_transform = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]) 
            first_transformed_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_origin.obj")
            mesh.apply_transform(rot_coordinate_transform)
            # mesh.apply_scale(scale_anysplat*setting_scale)
            # mesh_origin = copy.deepcopy(mesh)
            mesh.export(first_transformed_mesh_path)
            print(f"🟢 Saved object mesh whose coordinate aligned with gaussian's: {first_transformed_mesh_path}")

            # bbox = mesh.bounds
            # size = bbox[1] - bbox[0]
            # dx, dy, dz = size
            # print("X size:", dx)
            # print("Y size:", dy)
            # print("Z size:", dz)
            # print("Longest edge:", max(dx, dy, dz))

            # xyz = gs_origin.get_xyz
            # xyz_cv = xyz.clone()
            # xyz_cv = xyz_cv * (scale_anysplat*setting_scale)
            # gs_origin.from_xyz(xyz_cv)
            # adjust_scale = gs_origin.get_scaling * (scale_anysplat*setting_scale)
            # gs_origin.mininum_kernel_size *= (scale_anysplat*setting_scale)
            # gs_origin.from_scaling(adjust_scale)
            # single_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_resize.ply")
            # gs_origin.save_ply(single_ply_path)
            # print(f"🟢 Saved transformed object PLY: {single_ply_path}")

        # 如果显存很紧张，可以在这里 del single_scene / video 等
        # del single_scene

        # 显式释放显存
        del out
        torch.cuda.empty_cache()

    print("✅ All objects processed and saved as .pt")

if __name__ == "__main__":
    main()
