import os
import glob
import argparse
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from run_sam3d_multi import clean_name

from inference import (
    make_scene,
    transform_mesh,
    ready_gaussian_for_video_rendering,
    render_video,
    interactive_visualizer,
    _fix_gaussian_alignment
)
import trimesh
import pyrender
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import cv2

def compute_fov_from_intrinsics(focal_length, image_size, degrees=True):
    """
    从焦距和图像大小计算FOV
    
    Args:
        focal_length: 焦距，可以是单个值(假设fx=fy)或(fx, fy)元组
        image_size: 图像尺寸 (height, width)
        degrees: 是否以度为单位返回FOV
    
    Returns:
        fov: 垂直或水平视场角
    """
    # 参数校验
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size must be (height, width)")
    else:
        raise TypeError("image_size must be a tuple or list")
    
    if isinstance(focal_length, (tuple, list)):
        fx, fy = focal_length
    else:
        fx = fy = focal_length  # 假设方形像素，fx = fy
    
    height, width = image_size
    
    # 计算垂直FOV（以弧度为单位）
    fov_y_rad = 2 * np.arctan(height / (2 * fy))
    
    if degrees:
        fov_y = np.degrees(fov_y_rad)
    else:
        fov_y = fov_y_rad
    
    return fov_y

def mesh_rendering(mesh_path, extrinsics, fov_y, z_shift=0):
    mesh = trimesh.load(mesh_path)
    # 沿z轴移动
    mesh.vertices= mesh.vertices + np.array([0, 0, z_shift])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # 相机内参
    camera = pyrender.PerspectiveCamera(
        yfov=fov_y,
        aspectRatio=1.0
    )

    # 相机位姿（camera-to-world）
    # extrinsics：OpenCV world → camera
    T_wc = extrinsics

    # OpenCV camera → OpenGL camera
    cv_to_gl = np.array([
        [ 1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1],
    ])

    # pyrender 需要的是 camera → world
    camera_pose = np.linalg.inv(T_wc) @ cv_to_gl
    # camera_pose = np.linalg.inv(extrinsics)
    # camera_pose = extrinsics
    # camera_pose = np.eye(4)
    print('camera pose', camera_pose)
    scene.add(camera, pose=camera_pose)
    axis = trimesh.creation.axis(axis_length=0.5)
    scene.add(pyrender.Mesh.from_trimesh(axis,smooth=False))

    # 光源（很重要，否则是黑的）
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(448, 448)
    color, depth = renderer.render(scene)
    print(f'max depth, {depth.max()}, min depth, {depth.min()}, mean depth, {depth.mean()}, sum depth, {depth[depth>0].mean()}'   )
    renderer.delete()
    return color, depth

def mesh_rendering_with_depth_adjustment(mesh_path, extrinsics, fov_y, original_mean_depth, original_size):
    print(mesh_path)
    color, depth = mesh_rendering(mesh_path=mesh_path,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
    mean_depth_sam3d = np.mean(depth[depth > 0])
    print(f'mean depth sam3d, {mean_depth_sam3d}')
    z_shift = original_mean_depth-mean_depth_sam3d
    # 把 sam-3d结果放到跟anysplat背景相似的深度
    color, depth = mesh_rendering(mesh_path=mesh_path,extrinsics=extrinsics,fov_y=fov_y/180*np.pi,z_shift=z_shift)
    
    valid = depth > 0
    depth_fg = depth[valid]
    mean_depth = depth_fg.mean()
    min_depth = depth_fg.min()
    max_depth = depth_fg.max()
    print(f"mean depth: {mean_depth:.4f}")
    print(f"min depth:  {min_depth:.4f}")
    print(f"max depth:  {max_depth:.4f}")
    size_sam3d = np.sum(valid)
    scale = original_size/size_sam3d
    print(f'size sam3d = {size_sam3d}')
    print(f'size anysplat = {original_size}')
    print(f'scale = {scale}')
    # 如果需要打印统计信息
    print("Mean depth (excluding zeros):", np.mean(depth[~mask]))
    print("Mean depth (all valid pixels):", np.mean(depth[depth > 0]))
    return color, depth

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
    # image_name = os.path.basename(os.path.dirname(image_path))
     # 获取图像文件名（不包含扩展名），用于创建文件夹
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    # 清理文件名中的特殊字符
    image_name_clean = clean_name(image_name)
    # 创建图像名对应的文件夹
    image_output_dir = os.path.join(args.save_dir, image_name_clean)

    # 读取图像
    img1 = cv2.imread('/home/discover/sam3d_gs/bg-rgb/new-desk-o.jpg')
    mask_names = ['pot','bottle','duster']
    mask_paths = []
    mesh_paths = []
    masks = []
    original_size = []
    for i,item in enumerate(mask_names):
        mask_path = f'/home/discover/sam3d_gs/masks/new-desk/{item}.png'
        mask_paths.append(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = np.where(mask>0, 1, 0).astype(np.uint8)
        masks.append(valid_mask)
        original_size.append(np.sum(valid_mask))
        # mesh_path = f'/home/discover/sam3d_gs/masks/new-desk/newdesk_{item}-mesh_untransformed_trans.obj'
        mesh_path = f'/home/discover/sam3d_gs/sam-3d-objects/gaussians/single/newdesk_{item}-mesh-transformed.obj'
        mesh_paths.append(mesh_path)

    extrinsics = np.load('/home/discover/sam3d_gs/masks/new-desk/extrinsic.npy')
    intrinsics = np.load('/home/discover/sam3d_gs/masks/new-desk/intrinsic.npy')
    depth      = np.load('/home/discover/sam3d_gs/masks/new-desk/depth.npy')

    # 打印信息 (已修正标签错误)
    print("extrinsics", type(extrinsics), extrinsics) 
    print("intrinsics", type(intrinsics), intrinsics) # 修正了这里的字符串
    print("depth shape", depth.shape)

    image_size = (448, 448)

    # 1. 从归一化内参矩阵提取值
    fx_norm = intrinsics[0, 0]
    fy_norm = intrinsics[1, 1]

    # 2. 转换为像素焦距
    fx_pixels = fx_norm * (image_size[1] / 2.0)  # 乘以宽度/2
    fy_pixels = fy_norm * (image_size[0] / 2.0)  # 乘以高度/2

    print(f"归一化焦距: fx={fx_norm:.4f}, fy={fy_norm:.4f}")
    print(f"像素焦距: fx={fx_pixels:.2f}, fy={fy_pixels:.2f}")

    # 3. 使用像素焦距计算真实的FOV
    fov_y = compute_fov_from_intrinsics(fy_pixels, image_size, degrees=True)
    fov_x = compute_fov_from_intrinsics(fx_pixels, image_size, degrees=True)

    print(f"垂直FOV: {fov_y:.2f}度")
    print(f"水平FOV: {fov_x:.2f}度")

    camera_to_world = np.linalg.inv(extrinsics)  # shape (4, 4)

    rotation_mat = camera_to_world[:3, :3]
    translation_vec = camera_to_world[:3, 3]

    mean_depths_ori = []
    min_depths_ori = []
    max_depths_ori = []
    for i,item in enumerate(mask_names):
        print(f"{i} {item}")
        depth_fg = depth[masks[i]]

        mean_depth_ori = depth_fg.mean()
        mean_depths_ori.append(mean_depth_ori)
        min_depth_ori = depth_fg.min()
        min_depths_ori.append(min_depth_ori)
        max_depth_ori = depth_fg.max()
        max_depths_ori.append(max_depth_ori)

    # 这里不再限定 object_*.pt，而是把 save-dir/image_name 下所有 .pt 都吃掉
    paths = sorted(glob.glob(os.path.join(image_output_dir, "*.pt")))
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
        # out = torch.load(p, map_location=device)
        out = torch.load(p, map_location=device,weights_only=False)
        # 输出out 的dict键
        print(f"  Output keys: {list(out.keys())}")
        
        outputs.append(out)

        # 只用 make_scene，不做 ready_gaussian_for_video_rendering
        single_scene = make_scene(out)

        xyz = single_scene.get_xyz
        xyz_cv = xyz.clone()
        xyz_cv[:, 1] = -xyz[:, 1]  # Y轴翻转      
        xyz_cv[:, 0] = -xyz[:, 0]  # X轴翻转
        single_scene.from_xyz(xyz_cv)

        transformed_single_scene = _fix_gaussian_alignment(single_scene)

        stem = os.path.splitext(os.path.basename(p))[0]
        # single_ply_path = os.path.join(single_gauss_dir, f"{stem}.ply")
        single_ply_path = os.path.join(single_gauss_dir, f"{stem}_gs_target.ply")
        single_scene.save_ply(single_ply_path)
        print(f"🟢 Saved single-object PLY: {single_ply_path}")
        
        # 输出未经transform的gaussian
        untransforrmed_ply_path = os.path.join(single_gauss_dir, f"{stem}_gs_untransformed.ply")
        out["gs"].save_ply(untransforrmed_ply_path)
        print(f"🟢 Saved untransformed single-object PLY: {untransforrmed_ply_path}")
        
        # 输出 gs_2
        # single_ply_path_2 = os.path.join(single_gauss_dir, f"{stem}_gs_2.ply")
        # transformed_single_scene.save_ply(single_ply_path_2)
        
        # 打印object的pose，scale
        rotation = out['rotation']
        translation = out['translation']
        scale = out['scale']
        print(f" rotation: {out['rotation']}")
        print(f" translation: {out['translation']}")
        print(f" scale: {out['scale']}")            
        # print(f" gs rotation: {single_scene.get_rotation}")
        # print(f" gs scale: {single_scene.get_scaling}")      

        
        if out['glb']:
            mesh = out['glb']
            untransformed_mesh_path = os.path.join(single_gauss_dir, f"{stem}-mesh_untransformed.obj")
            mesh.export(untransformed_mesh_path)
            print(f"🟢 Saved untransformed object Mesh: {untransformed_mesh_path}")
            
            # trimesh坐标系（x右，y前，z下）转向目标坐标系 （x右，y下，z前）
            rot_coordinate_transform = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]) 
            first_transformed_mesh_path = os.path.join(single_gauss_dir, f"{stem}-mesh_cooridnates-adjusted.obj")
            mesh.apply_transform(rot_coordinate_transform)
            mesh.export(first_transformed_mesh_path)
            print(f"🟢 Saved untransformed object Mesh: {first_transformed_mesh_path}")
            
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
            # x, y轴取反
            mesh.vertices[:,1] = -mesh.vertices[:,1]
            mesh.vertices[:,0] = -mesh.vertices[:,0]
            # single_mesh_path = os.path.join(single_gauss_dir, f"{stem}-yz-inverse3.obj")
            single_mesh_path = os.path.join(single_gauss_dir, f"{stem}-mesh-transformed.obj")
            mesh.export(single_mesh_path)
            print(f"🟢 Saved single-object Mesh: {single_mesh_path}")

        # if args.export_gif:
        #     video = render_video(
        #         single_scene,
        #         r=1,
        #         fov=60,
        #         resolution=512,
        #     )["color"]

        #     single_gif_path = os.path.join(single_gauss_dir, f"{stem}.gif")
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
    # scene_gs = make_scene(*outputs)
    # scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    # gauss_dir = os.path.join(project_root, "gaussians", "multi")
    # os.makedirs(gauss_dir, exist_ok=True)

    # ply_path = os.path.join(gauss_dir, f"{image_name}.ply")
    # scene_gs.save_ply(ply_path)
    # print(f"✅ Saved merged PLY: {ply_path}")

    # if args.export_gif:
    #     video = render_video(
    #         scene_gs,
    #         r=1,
    #         fov=60,
    #         resolution=512,
    #     )["color"]

    #     gif_path = os.path.join(gauss_dir, f"{image_name}.gif")
    #     imageio.mimsave(
    #         gif_path,
    #         video,
    #         format="GIF",
    #         duration=1000 / 30,  # 30fps
    #         loop=0,
    #     )
    #     print(f"✅ Saved merged GIF: {gif_path}")


if __name__ == "__main__":
    main()
