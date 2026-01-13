import re
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from PIL import Image
import trimesh
import pyrender
import numpy as np

def clean_name(x: str):
    return re.sub(r'[^0-9a-zA-Z_-]', '', x)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def collect_mask_paths(mask_root: str):
    """
    递归收集 mask_root 下所有 png/jpg/jpeg 的路径。
    """
    all_mask_paths = []
    for root, _, files in os.walk(mask_root):
        for f in files:
            lf = f.lower()
            if lf.endswith(".png") or lf.endswith(".jpg") or lf.endswith(".jpeg"):
                all_mask_paths.append(os.path.join(root, f))

    all_mask_paths.sort()
    print(f"Found {len(all_mask_paths)} mask files under {mask_root}")
    return all_mask_paths


def load_binary_mask(path: str):
    """
    单个 mask 文件 → 二值 uint8 数组 (H, W), {0, 1}
    """
    m = np.array(Image.open(path).convert("L"))
    m = (m > 128).astype("uint8")
    return m

def compute_fov_from_intrinsics(fx, fy, image_size, degrees=True):
    """
    从像素单位的 fx, fy 计算水平 / 垂直 FOV
    """
    height, width = image_size

    fov_y = 2 * np.arctan(height / (2 * fy))
    fov_x = 2 * np.arctan(width  / (2 * fx))

    if degrees:
        fov_y = np.degrees(fov_y)
        fov_x = np.degrees(fov_x)

    return fov_x, fov_y

def mesh_rendering(mesh, extrinsics, fov_y):
    # mesh = trimesh.load(mesh_path)
    # 沿z轴移动
    # mesh.vertices= mesh.vertices + np.array([0, 0, z_shift])
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
    # print('camera pose', camera_pose)
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

def mesh_rendering_with_depth_adjustment(mesh, extrinsics, fov_y, original_mean_depth, original_size):
    # print(mesh_path)
    mesh_copy = mesh.copy()
    color, depth = mesh_rendering(mesh=mesh,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
    mean_depth_sam3d = np.mean(depth[depth > 0])
    print(f'mean depth sam3d, {mean_depth_sam3d}')
    z_shift = original_mean_depth-mean_depth_sam3d
    # 把 sam-3d结果放到跟anysplat背景相似的深度
    color, depth = mesh_rendering(mesh=mesh_copy,extrinsics=extrinsics,fov_y=fov_y/180*np.pi)
    
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
    return color, depth, scale