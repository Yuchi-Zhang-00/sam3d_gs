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
from PIL import Image
import cv2

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3D multi-object inference and save outputs to .pt, and reconstruct single object Gaussian .ply"
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
        help="Original image path (Input image path to lift to 3D.).",
    )
    # parser.add_argument(
    #     "--mask-root",
    #     type=str,
    #     default="sam3/agent_output_multi/masks",
    #     help="Directory containing mask PNG/JPGs.",
    # )
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

    # 获取图像文件名（不包含扩展名），用于创建文件夹
    # image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # 清理文件名中的特殊字符
    # image_name_clean = clean_name(image_name)
    # 创建图像名对应的文件夹
    # image_output_dir = os.path.join(args.save_dir, image_name_clean)

    # os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    extrinsics = np.load(os.path.join(image_dir, 'extrinsic.npy'))
    intrinsics = np.load(os.path.join(image_dir, 'intrinsic.npy'))
    depth_anysplat      = np.load(os.path.join(image_dir, 'depth.npy'))

    # 打印信息 (已修正标签错误)
    print("extrinsics", type(extrinsics), extrinsics) 
    print("intrinsics", type(intrinsics), intrinsics) # 修正了这里的字符串
    print("depth_anysplat shape", depth_anysplat.shape)

    image_size = (448, 448)

    # # 1. 从归一化内参矩阵提取值
    # fx_norm = intrinsics[0, 0]
    # fy_norm = intrinsics[1, 1]

    # # 2. 转换为像素焦距
    # fx_pixels = fx_norm * (image_size[1] / 2.0)  # 乘以宽度/2
    # fy_pixels = fy_norm * (image_size[0] / 2.0)  # 乘以高度/2

    fx_pixels = intrinsics[0, 0]
    fy_pixels = intrinsics[1, 1]

    # print(f"归一化焦距: fx={fx_norm:.4f}, fy={fy_norm:.4f}")
    print(f"像素焦距: fx={fx_pixels:.2f}, fy={fy_pixels:.2f}")

    # 3. 使用像素焦距计算真实的FOV
    fov_x, fov_y = compute_fov_from_intrinsics(fx_pixels, fy_pixels, image_size, degrees=True)

    print(f"垂直FOV: {fov_y:.2f}度")
    print(f"水平FOV: {fov_x:.2f}度")

    camera_to_world = np.linalg.inv(extrinsics)  # shape (4, 4)

    rotation_mat = camera_to_world[:3, :3]
    translation_vec = camera_to_world[:3, 3]

    mean_depths_ori = []
    min_depths_ori = []
    max_depths_ori = []
    # for i,item in enumerate(mask_names):
    #     print(f"{i} {item}")
    #     depth_fg = depth[masks[i]]

    #     mean_depth_ori = depth_fg.mean()
    #     mean_depths_ori.append(mean_depth_ori)
    #     min_depth_ori = depth_fg.min()
    #     min_depths_ori.append(min_depth_ori)
    #     max_depth_ori = depth_fg.max()
    #     max_depths_ori.append(max_depth_ori)

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
        # print(depth_fg)
        mean_depth_ori = depth_fg.mean()
        min_depth_ori = depth_fg.min()
        max_depth_ori = depth_fg.max()
        print(f"anyplat mean_depth_ori: {mean_depth_ori:.4f}, min_depth_ori: {min_depth_ori:.4f}, max_depth_ori: {max_depth_ori:.4f}")
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

        # # 输出out 的dict键
        # print(f"  Output keys: {list(out.keys())}")

        # 根据out的位姿将Gaussian转换到sam3d估计的位姿
        single_scene = make_scene(out)
        # 改动x，z轴朝向与anysplat的结果对齐 （x右y下z前）
        xyz = single_scene.get_xyz
        xyz_cv = xyz.clone()
        xyz_cv[:, 1] = -xyz[:, 1]  # Y轴翻转      
        xyz_cv[:, 0] = -xyz[:, 0]  # X轴翻转
        single_scene.from_xyz(xyz_cv)

        # stem = os.path.splitext(os.path.basename(p))[0]
        # single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
        single_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_sam3d_target.ply")
        single_scene.save_ply(single_ply_path)
        # print(f"🟢 Saved single-object PLY: {single_ply_path}")
        
        # 输出未经transform的Gaussian
        untransforrmed_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_untransformed.ply")
        # out["gs"].save_ply(untransforrmed_ply_path)
        print(f"🟢 Saved untransformed single-object PLY: {untransforrmed_ply_path}")
        
        # 打印object的pose，scale
        rotation = out['rotation']
        translation = out['translation']
        scale = out['scale']
        # print(f" rotation: {out['rotation']}")
        # print(f" translation: {out['translation']}")
        # print(f" scale: {out['scale']}")            
        
        if out['glb']:              # 如果输出包含mesh
            mesh = out['glb']
            untransformed_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_untransformed.obj")
            # mesh.export(untransformed_mesh_path)
            print(f"🟢 Saved untransformed object Mesh: {untransformed_mesh_path}")
            
            # 输出的mesh在trimesh坐标系下，将其转向输出Gaussian的初始坐标系 
            rot_coordinate_transform = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]) 
            first_transformed_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_cooridnates_aligned.obj")
            mesh.apply_transform(rot_coordinate_transform)
            # mesh.export(first_transformed_mesh_path)
            print(f"🟢 Saved object mesh whose coordinate aligned with gaussian's: {first_transformed_mesh_path}")
            
            # 1. 处理旋转：四元数 → 旋转矩阵
            quat = out["rotation"].cpu().numpy()  # pytorch3d中四元数[w, x, y, z]
            rot = R.from_quat(quat,scalar_first=True).as_matrix().squeeze(0) # 3x3

            inverse_rot = np.linalg.inv(rot)
            # 2. 处理缩放
            scale = out["scale"].squeeze(0).cpu().numpy() 
            if np.isscalar(scale):
                scale = np.array([scale, scale, scale])
            else:
                scale = np.asarray(scale)
            
            # 构建缩放矩阵（3x3）
            scale_mat = np.diag(scale)

            # 3. 组合旋转 + 缩放：先缩放，再旋转（通常顺序）
            # 即：R @ S （对点 p：p' = R @ (S @ p) = (R @ S) @ p）
            # rot_scale = rot @ scale_mat  # 3x3  

            # 4. 构建 4x4 齐次变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = inverse_rot @ scale_mat
            transform[:3, 3] = out["translation"].cpu().numpy() 

            # 5. 应用变换到 mesh
            mesh.apply_transform(transform)
            # x, y轴取反，朝向目标坐标系（x右，y下，z前）
            mesh.vertices[:,1] = -mesh.vertices[:,1]
            mesh.vertices[:,0] = -mesh.vertices[:,0]
            # single_mesh_path = os.path.join(single_gauss_dir, f"{stem}-yz-inverse3.obj")
            single_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_sam3d_taget.obj")
            mesh.export(single_mesh_path)
            # print(f"🟢 Saved single-object Mesh: {single_mesh_path}")
            # color, depth, scale, z_shift = mesh_rendering_with_depth_adjustment(mesh=mesh, extrinsics=extrinsics, fov_y=fov_y, original_mean_depth=mean_depth_ori,original_size=size_ori)
            mesh_copy = mesh.copy()
            color, depth = mesh_rendering(mesh=mesh_copy,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
            mean_depth_sam3d = np.mean(depth[depth > 0])
            # z_shift = mean_depth_ori - mean_depth_sam3d
            z_shift = min_depth_ori - mean_depth_sam3d
            print(f" z_shift:{ z_shift}, mean_depth_sam3d: {mean_depth_sam3d}, mean_depth_ori: {mean_depth_ori}")
            mesh.vertices= mesh.vertices + np.array([0, 0, z_shift])
            mesh_copy = mesh.copy()
            color, depth = mesh_rendering(mesh=mesh_copy,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
            depth_fg = depth[depth > 0]
            size_new = np.sum(depth > 0)
            scale = size_ori/size_new
            print(f"mask_stem: {mask_stem}", "size_ori:", size_ori, "size_new:", size_new, "scale:", scale, "z_shift:", z_shift)
            mesh.vertices = mesh.vertices * scale
            mesh_copy = mesh.copy()
            color, depth = mesh_rendering(mesh=mesh_copy,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
            mean_depth_sam3d_2 = np.mean(depth[depth > 0])
            z_shift_2 = mean_depth_ori - mean_depth_sam3d_2
            # z_shift_2 = min_depth_ori - mean_depth_sam3d_2
            mesh.vertices= mesh.vertices + np.array([0, 0, z_shift_2])
            # mesh已经根据z_shift和scale调整过了，现在将Gaussian进行同样的调整
            xyz = single_scene.get_xyz
            xyz_cv = xyz.clone()
            xyz_cv[:, 2] = xyz[:, 2] + z_shift  # 在Z方向上第一次移动
            xyz_cv = xyz_cv * scale
            # xyz_cv[:, 2] = xyz[:, 2] + z_shift_2  # 在Z方向上第二次移动   因为mesh移动了两次
            single_scene.from_xyz(xyz_cv)
            adjust_scale = single_scene.get_scaling * scale
            single_scene.mininum_kernel_size *= scale
            single_scene.from_scaling(adjust_scale)
            xyz = single_scene.get_xyz
            xyz_cv = xyz.clone()
            xyz_cv[:, 2] = xyz[:, 2] + z_shift_2  # 在Z方向上第二次移动   因为mesh移动了两次
            single_scene.from_xyz(xyz_cv)
            single_ply_path = os.path.join(assets_dir, f"{mask_stem}_gs_final.ply")
            single_scene.save_ply(single_ply_path)
            print(f"🟢 Saved transformed object PLY: {single_ply_path}")

            transformed_mesh_path = os.path.join(assets_dir, f"{mask_stem}_mesh_final.obj")
            mesh.export(transformed_mesh_path)
            print(f"🟢 Saved transformed object mesh: {transformed_mesh_path}")

            





        # 如果显存很紧张，可以在这里 del single_scene / video 等
        del single_scene

        # 显式释放显存
        del out
        torch.cuda.empty_cache()

    print("✅ All objects processed and saved as .pt")

    # project_root = args.project_root
    # image_path = args.image_path
    # image_name = os.path.basename(os.path.dirname(image_path))

    # 读取图像
    # img1 = cv2.imread('/home/discover/sam3d_gs/bg-rgb/new-desk-o.jpg')
    # mask_names = ['pot','bottle','duster']
    # mask_paths = []
    # mesh_paths = []
    # masks = []
    # original_size = []
    # for i,item in enumerate(mask_names):
    #     mask_path = f'/home/discover/sam3d_gs/masks/new-desk/{item}.png'
    #     mask_paths.append(mask_path)
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     valid_mask = np.where(mask>0, 1, 0).astype(np.uint8)
    #     masks.append(valid_mask)
    #     original_size.append(np.sum(valid_mask))
    #     # mesh_path = f'/home/discover/sam3d_gs/masks/new-desk/newdesk_{item}-mesh_untransformed_trans.obj'
    #     mesh_path = f'/home/discover/sam3d_gs/sam-3d-objects/gaussians/single/newdesk_{item}-mesh-transformed.obj'
    #     mesh_paths.append(mesh_path)

    

    # # 这里不再限定 object_*.pt，而是把 save-dir/image_name 下所有 .pt 都吃掉
    # paths = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    # if not paths:
    #     raise RuntimeError(f"No .pt found under {args.save_dir}")

    # print(f"Found {len(paths)} .pt files:")
    # for p in paths:
    #     print("  ", p)

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # =========================
    # # 1️⃣ 遍历每个 .pt：导出单物体 PLY + OBJ
    # # =========================
    # for idx, p in enumerate(paths):
    #     print(f"[{idx+1}/{len(paths)}] loading {p}")
    #     out = torch.load(p, map_location=device,weights_only=False)
    #     # 输出out 的dict键
    #     print(f"  Output keys: {list(out.keys())}")

    #     # 只用 make_scene，不做 ready_gaussian_for_video_rendering
    #     single_scene = make_scene(out)

    #     xyz = single_scene.get_xyz
    #     xyz_cv = xyz.clone()
    #     xyz_cv[:, 1] = -xyz[:, 1]  # Y轴翻转      
    #     xyz_cv[:, 0] = -xyz[:, 0]  # X轴翻转
    #     single_scene.from_xyz(xyz_cv)

    #     stem = os.path.splitext(os.path.basename(p))[0]
    #     # single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
    #     single_ply_path = os.path.join(assets_dir, f"{stem}_gs_target.ply")
    #     single_scene.save_ply(single_ply_path)
    #     print(f"🟢 Saved single-object PLY: {single_ply_path}")
        
    #     # 输出未经transform的gaussian
    #     untransforrmed_ply_path = os.path.join(assets_dir, f"{stem}_gs_untransformed.ply")
    #     out["gs"].save_ply(untransforrmed_ply_path)
    #     print(f"🟢 Saved untransformed single-object PLY: {untransforrmed_ply_path}")
        
    #     # 打印object的pose，scale
    #     rotation = out['rotation']
    #     translation = out['translation']
    #     scale = out['scale']
    #     print(f" rotation: {out['rotation']}")
    #     print(f" translation: {out['translation']}")
    #     print(f" scale: {out['scale']}")            
        
    #     if out['glb']:
    #         mesh = out['glb']
    #         untransformed_mesh_path = os.path.join(assets_dir, f"{stem}_mesh_untransformed.obj")
    #         mesh.export(untransformed_mesh_path)
    #         print(f"🟢 Saved untransformed object Mesh: {untransformed_mesh_path}")
            
    #         # trimesh坐标系（x右，y前，z下）转向目标坐标系 （x右，y下，z前）
    #         rot_coordinate_transform = np.array([
    #             [1, 0, 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 1, 0, 0],
    #             [0, 0, 0, 1]
    #         ]) 
    #         first_transformed_mesh_path = os.path.join(assets_dir, f"{stem}_mesh_cooridnates_aligned.obj")
    #         mesh.apply_transform(rot_coordinate_transform)
    #         mesh.export(first_transformed_mesh_path)
    #         print(f"🟢 Saved object mesh whose coordinate aligned with gaussian's: {first_transformed_mesh_path}")
            
    #         # 1. 处理旋转：四元数 → 旋转矩阵
    #         quat = out["rotation"].cpu().numpy()  # pytorch3d中四元数[w, x, y, z]
    #         rot = R.from_quat(quat,scalar_first=True).as_matrix().squeeze(0) # 3x3

    #         inverse_rot = np.linalg.inv(rot)
    #         # 2. 处理缩放
    #         scale = out["scale"].squeeze(0).cpu().numpy() 
    #         if np.isscalar(scale):
    #             scale = np.array([scale, scale, scale])
    #         else:
    #             scale = np.asarray(scale)
            
    #         # 构建缩放矩阵（3x3）
    #         scale_mat = np.diag(scale)

    #         # 3. 组合旋转 + 缩放：先缩放，再旋转（通常顺序）
    #         # 即：R @ S （对点 p：p' = R @ (S @ p) = (R @ S) @ p）
    #         # rot_scale = rot @ scale_mat  # 3x3  

    #         # 4. 构建 4x4 齐次变换矩阵
    #         transform = np.eye(4)
    #         transform[:3, :3] = inverse_rot @ scale_mat
    #         transform[:3, 3] = out["translation"].cpu().numpy() 

    #         # 5. 应用变换到 mesh
    #         mesh.apply_transform(transform)
    #         # x, y轴取反
    #         mesh.vertices[:,1] = -mesh.vertices[:,1]
    #         mesh.vertices[:,0] = -mesh.vertices[:,0]
    #         # single_mesh_path = os.path.join(single_gauss_dir, f"{stem}-yz-inverse3.obj")
    #         single_mesh_path = os.path.join(assets_dir, f"{stem}_mesh_transformed.obj")
    #         mesh.export(single_mesh_path)
    #         print(f"🟢 Saved single-object Mesh: {single_mesh_path}")

    #     # 如果显存很紧张，可以在这里 del single_scene / video 等
    #     del single_scene

    print("✅ All single-object scenes exported.")

if __name__ == "__main__":
    main()
