import re
import os
import atexit
os.environ["PYOPENGL_PLATFORM"] = "egl"
from PIL import Image
import trimesh
import pyrender
import numpy as np
import imageio


_DEFAULT_MESH_RENDERERS = {}


class MeshRenderContext:
    def __init__(
        self,
        width=448,
        height=448,
        add_axis=False,
        debug_depth_path=None,
        verbose=False,
    ):
        self.width = width
        self.height = height
        self.add_axis = add_axis
        self.debug_depth_path = debug_depth_path
        self.verbose = verbose
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        self.cv_to_gl = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def render(self, mesh, extrinsics, fov_y):
        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)

        if self.verbose:
            print(
                f"vertices shape {mesh.vertices.shape} "
                f"mesh vertices mean {np.mean(mesh.vertices, axis=0)}"
            )

        render_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            material=self.material,
            smooth=False,
        )

        scene = pyrender.Scene()
        scene.add(render_mesh)

        camera = pyrender.PerspectiveCamera(
            yfov=fov_y,
            aspectRatio=self.width / self.height,
        )

        camera_pose = extrinsics @ self.cv_to_gl
        scene.add(camera, pose=camera_pose)

        if self.add_axis:
            axis = trimesh.creation.axis(axis_length=0.5)
            scene.add(pyrender.Mesh.from_trimesh(axis, smooth=False))

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        color, depth = self.renderer.render(scene)

        if self.debug_depth_path:
            depth_min = depth.min()
            depth_range = depth.max() - depth_min
            if depth_range > 0:
                depth_normalized = (
                    (depth - depth_min) / depth_range * 255
                ).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            imageio.imwrite(self.debug_depth_path, depth_normalized)

        if self.verbose:
            valid_depth = depth[depth > 0]
            valid_mean = valid_depth.mean() if valid_depth.size > 0 else np.nan
            print(
                f"max depth {depth.max()}, min depth {depth.min()}, "
                f"mean depth {depth.mean()}, valid mean depth {valid_mean}"
            )

        return color, depth


def get_default_mesh_renderer(
    width=448,
    height=448,
    add_axis=False,
    debug_depth_path=None,
    verbose=False,
):
    key = (width, height, add_axis, debug_depth_path, verbose)
    renderer = _DEFAULT_MESH_RENDERERS.get(key)
    if renderer is None:
        renderer = MeshRenderContext(
            width=width,
            height=height,
            add_axis=add_axis,
            debug_depth_path=debug_depth_path,
            verbose=verbose,
        )
        _DEFAULT_MESH_RENDERERS[key] = renderer
    return renderer


def close_default_mesh_renderers():
    for renderer in _DEFAULT_MESH_RENDERERS.values():
        renderer.close()
    _DEFAULT_MESH_RENDERERS.clear()


atexit.register(close_default_mesh_renderers)


def clean_name(x: str):
    return re.sub(r'[^0-9a-zA-Z_-]', '', x)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def collect_mask_paths(mask_root: str):
    """Recursively collect all .png / .jpg / .jpeg paths under mask_root."""
    all_mask_paths = []
    for root, _, files in os.walk(mask_root):
        for f in files:
            lf = f.lower()
            if lf.endswith(".png") or lf.endswith(".jpg") or lf.endswith(".jpeg"):
                all_mask_paths.append(os.path.join(root, f))

    all_mask_paths.sort()
    print(f"Found {len(all_mask_paths)} mask files under {mask_root}")
    return all_mask_paths


def compute_fov_from_intrinsics(fx, fy, image_size, degrees=True):
    """Compute horizontal / vertical FOV from pixel-unit fx, fy."""
    height, width = image_size

    fov_y = 2 * np.arctan(height / (2 * fy))
    fov_x = 2 * np.arctan(width  / (2 * fx))

    if degrees:
        fov_y = np.degrees(fov_y)
        fov_x = np.degrees(fov_x)

    return fov_x, fov_y

def mesh_rendering(
    mesh,
    extrinsics,
    fov_y,
    renderer=None,
    width=448,
    height=448,
    add_axis=False,
    debug_depth_path=None,
    verbose=False,
):
    if renderer is None:
        renderer = get_default_mesh_renderer(
            width=width,
            height=height,
            add_axis=add_axis,
            debug_depth_path=debug_depth_path,
            verbose=verbose,
        )
    return renderer.render(mesh, extrinsics, fov_y)

