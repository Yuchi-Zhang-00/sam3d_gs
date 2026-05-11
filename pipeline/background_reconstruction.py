"""Batch RANSAC-based table alignment + 3D Gaussian export on top of AnySplat.

This is a cleaned-up rewrite of `submodule/AnySplat/inference_ransac_batch.py`.
The script now lives outside the AnySplat submodule, so it explicitly inserts
the AnySplat root onto `sys.path` to keep the original imports working.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

# ===== Make AnySplat's `src.*` and `utils.py` importable when running from the
# repository root (this file no longer lives inside submodule/AnySplat).
_ANYSPLAT_ROOT = Path(__file__).resolve().parent.parent / "submodule" / "AnySplat"
sys.path.insert(0, str(_ANYSPLAT_ROOT))
sys.path.insert(0, str(_ANYSPLAT_ROOT.parent))  # mirrors original sys.path entry

from src.misc.image_io import save_interpolated_video  # noqa: E402, F401
from src.model.ply_export import export_ply  # noqa: E402
from src.model.model.anysplat import AnySplat  # noqa: E402
from src.utils.image import process_image  # noqa: E402
from utils import (  # noqa: E402
    align_points_to_table,
    depth_to_points,
    fit_plane_ransac_safe_2,
    plane_coordinate_system,
    render_depth_from_points,
    shrink_mask_erode,
)


# ===== RANSAC / inner-rectangle hyperparameters =====
RANSAC_NUM_ITERS = 600
RANSAC_DIST_THRESH = 0.005  # tabletops are usually very flat
RANSAC_SAMPLE_N = 40000
INNER_PERCENTILE = (20, 80)  # crop to the central 60% to avoid edges
MIN_INNER_POINTS = 50

# ===== Scene normalisation =====
# Quantile of |xyz| used as the reference radius before rescaling, and the
# target radius the reference is mapped to.
SCALE_QUANTILE = 0.95
SCALE_TARGET_RANGE = 0.6

# ===== Post-alignment scene placement =====
# Offsets applied after table-alignment so the aligned cloud can be shifted
# from the origin if the downstream consumer needs it (e.g. to place it on a
# Mujoco table). Defaults are 0, meaning the aligned cloud sits at the origin.
DEFAULT_X_OFFSET = 0.0
DEFAULT_Z_OFFSET = 0.0

# ===== Mask shrink before plane fitting =====
BG_MASK_SHRINK_RATIO = 0.12

# ===== Default model id =====
DEFAULT_MODEL_ID = "lhjiang/anysplat"


def compute_table_geometry_ransac(depth, mask, intrinsic, extrinsic):
    """Fit a tabletop plane via RANSAC + inner PCA and build a world-aligned
    transform that maps the original world frame onto a table-aligned frame.
    """
    H, W = depth.shape

    # ===== 1. Intrinsics =====
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # ===== 2. Depth -> camera-frame points =====
    points_cam = depth_to_points(depth, mask, fx, fy, cx, cy)
    print("points_cam:", points_cam.shape)

    # ===== 3. RANSAC plane =====
    normal_cam, center_cam, inlier_idx = fit_plane_ransac_safe_2(
        points_cam,
        num_iters=RANSAC_NUM_ITERS,
        dist_thresh=RANSAC_DIST_THRESH,
        sample_N=RANSAC_SAMPLE_N,
    )
    print(f"RANSAC normal: {normal_cam}")

    pts_plane = points_cam[inlier_idx]

    # ===== 4. Plane coordinate system =====
    u, v = plane_coordinate_system(normal_cam)
    rel = pts_plane - center_cam
    pts_2d = np.stack([rel @ u, rel @ v], axis=1)

    # ===== 5. Inner rectangle (crop edges) =====
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    x_min, x_max = np.percentile(x, list(INNER_PERCENTILE))
    y_min, y_max = np.percentile(y, list(INNER_PERCENTILE))
    inner = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    pts_inner = pts_2d[inner]
    if pts_inner.shape[0] < MIN_INNER_POINTS:
        raise RuntimeError("Too few inner RANSAC points")

    # ===== 6. PCA on the inner points =====
    mean_2d = pts_inner.mean(axis=0)
    centered = pts_inner - mean_2d
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    dir_long_2d = Vt[0]

    # ===== 7. 2D -> 3D =====
    dir_long_cam = dir_long_2d[0] * u + dir_long_2d[1] * v
    dir_long_cam /= np.linalg.norm(dir_long_cam)
    dir_short_cam = np.cross(normal_cam, dir_long_cam)
    dir_short_cam /= np.linalg.norm(dir_short_cam)

    # ===== 8. World consistency (avoid axis flip) =====
    R_cw = extrinsic[:3, :3]
    if (R_cw @ dir_long_cam)[0] < 0:
        dir_long_cam = -dir_long_cam
        dir_short_cam = -dir_short_cam

    # ===== 9. OBB extents =====
    proj = centered @ Vt[:2].T
    min_xy, max_xy = proj.min(0), proj.max(0)
    length = max_xy[0] - min_xy[0]
    width = max_xy[1] - min_xy[1]

    center_plane_cam = center_cam + mean_2d[0] * u + mean_2d[1] * v

    # ===== 10. Build world->table alignment =====
    R_table_cam = np.stack([dir_long_cam, dir_short_cam, normal_cam], axis=1)
    R_align_cam = R_table_cam.T
    t_align_cam = -R_align_cam @ center_plane_cam

    R_align_world = R_align_cam @ R_cw
    t_align_world = R_align_cam @ extrinsic[:3, 3] + t_align_cam

    print("RANSAC inlier ratio:", len(inlier_idx) / points_cam.shape[0])

    return {
        "length": float(length),
        "width": float(width),
        "normal": normal_cam,
        "dir_long": dir_long_cam,
        "dir_short": dir_short_cam,
        "R_align_cam": R_align_cam,
        "t_align_cam": t_align_cam,
        "R_align_world": R_align_world,
        "t_align_world": t_align_world,
    }


def _save_depth_npy_and_viz(depth, image_folder, base_name):
    """Save a raw depth array and a normalized 8-bit visualisation."""
    depth_path = Path(image_folder) / f"{base_name}.npy"
    np.save(depth_path, depth)
    viz = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    viz_path = Path(image_folder) / f"{base_name}_visual.png"
    imageio.imwrite(viz_path, viz)


def process_single_image(image_path, model, device, args):
    """Run AnySplat on one `clean_background.png` and export aligned assets."""
    image_folder = os.path.dirname(image_path)
    image_ori_path = os.path.join(image_folder, "input_image.png")

    # Load images.
    image = process_image(image_path)
    image_ori = process_image(image_ori_path)
    images_ori = torch.stack([image_ori], dim=0).unsqueeze(0).to(device)
    images = torch.stack([image], dim=0).unsqueeze(0).to(device)
    b, v, _, H, W = images.shape

    # Inference.
    with torch.no_grad():
        gaussians, pred_context_pose, depth_dict = model.inference((images + 1) * 0.5)
        gaussians_ori, pred_context_pose_ori, depth_dict_ori = model.inference(
            (images_ori + 1) * 0.5
        )
    depth_ori = depth_dict_ori["depth"][0][0].squeeze().cpu().numpy()
    _save_depth_npy_and_viz(depth_ori, image_folder, "depth_ori")

    # Camera parameters. AnySplat returns camera-to-world; we store world-to-camera.
    pred_all_extrinsic = pred_context_pose["extrinsic"][0][0].inverse().cpu().numpy()
    pred_all_intrinsic = pred_context_pose["intrinsic"][0][0].cpu().numpy()
    print(f"Processing {os.path.basename(image_folder)}: converted intrinsics:")
    print(
        f"  fx: {pred_all_intrinsic[0, 0] * W:.2f}, "
        f"fy: {pred_all_intrinsic[1, 1] * H:.2f}"
    )
    print(
        f"  cx: {pred_all_intrinsic[0, 2] * W:.2f}, "
        f"cy: {pred_all_intrinsic[1, 2] * H:.2f}"
    )

    # Scale normalised intrinsics to pixel units.
    pred_all_intrinsic[0, :] = pred_all_intrinsic[0, :] * W
    pred_all_intrinsic[1, :] = pred_all_intrinsic[1, :] * H

    np.save(Path(image_folder) / "extrinsic.npy", pred_all_extrinsic)
    np.save(Path(image_folder) / "intrinsic.npy", pred_all_intrinsic)

    intrinsic = pred_all_intrinsic
    extrinsic = pred_all_extrinsic
    gaussian_xyz = gaussians.means[0].detach().cpu().numpy()
    depth = depth_dict["depth"][0][0].squeeze().cpu().numpy()
    _save_depth_npy_and_viz(depth, image_folder, "depth")

    # Asset directory.
    assets_folder = os.path.join(image_folder, "3d_assets")
    os.makedirs(assets_folder, exist_ok=True)

    # Export the raw 3DGS reconstruction.
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(assets_folder) / "bg.ply",
    )

    if not args.align_table:
        print(
            "Table alignment disabled (--no-align-table); "
            "skipping bg_aligned.ply export."
        )
        print(f"Done. Outputs saved under: {image_folder}")
        return

    # Re-render depth from the splat point cloud (used for plane fitting).
    depth_point = render_depth_from_points(gaussian_xyz, intrinsic, extrinsic, H, W)

    mask_path = Path(image_folder) / "bg_mask.png"
    if not mask_path.exists():
        print(f"Warning: bg_mask.png not found, skipping table alignment: {mask_path}")
        return

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    mask = shrink_mask_erode(mask, ratio=BG_MASK_SHRINK_RATIO)

    result = compute_table_geometry_ransac(
        depth=depth_point,
        mask=mask,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    print(f"\n{os.path.basename(image_folder)} table geometry:")
    print(f"  length (m): {result['length']:.3f}")
    print(f"  width  (m): {result['width']:.3f}")
    print(f"  normal: {result['normal']}")

    # Align the splat point cloud to the table frame.
    points_table_world = align_points_to_table(
        gaussian_xyz,
        result["R_align_world"],
        result["t_align_world"],
    )
    points_table_world = points_table_world - np.median(points_table_world, axis=0)

    # Use a robust quantile for scale so outliers don't dominate.
    abs_points = np.abs(points_table_world)
    ref_range = np.quantile(abs_points, SCALE_QUANTILE)
    scale_factor = ref_range / SCALE_TARGET_RANGE
    points_table_world = points_table_world / scale_factor
    gaussians.scales[0] = gaussians.scales[0] / scale_factor

    np.save(Path(image_folder) / "scale.npy", scale_factor)
    print(f"  scale factor: {scale_factor:.3f}")

    # Swap X/Y, flip Z, then apply optional placement offsets (default 0,0).
    x = points_table_world[:, 0].copy()
    y = points_table_world[:, 1].copy()
    points_table_world[:, 0] = y
    points_table_world[:, 1] = x
    points_table_world[:, 2] *= -1
    points_table_world[:, 2] += args.z_offset
    points_table_world[:, 0] += args.x_offset

    export_ply(
        points_table_world,
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(assets_folder) / "bg_aligned.ply",
    )

    print(
        f"  Z range: min={points_table_world[:, 2].min():.3f}, "
        f"max={points_table_world[:, 2].max():.3f}"
    )
    print(f"Done. Outputs saved under: {image_folder}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a 3D Gaussian model from a single image and emit the "
            "associated camera intrinsics/extrinsics, depth maps, and an "
            "optional table-aligned point cloud."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory or single file. Directories are searched recursively for clean_background.{png,jpg}.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id to load (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--align-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run RANSAC table alignment and export bg_aligned.ply. "
            "Use --no-align-table to disable (only bg.ply will be emitted). "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=DEFAULT_X_OFFSET,
        help="X-axis offset (m) applied after table alignment. Default: 0 (origin).",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=DEFAULT_Z_OFFSET,
        help="Z-axis offset (m) applied after table alignment. Default: 0 (origin).",
    )

    args = parser.parse_args()

    if os.path.isfile(args.input_dir):
        input_dir = os.path.dirname(args.input_dir)
    else:
        input_dir = args.input_dir

    print(f"Loading model: {args.model_id}")
    model = AnySplat.from_pretrained(args.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded.")

    clean_background_files = []
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower() in ("clean_background.png", "clean_background.jpg"):
                clean_background_files.append(os.path.join(root, file))

    print(f"Found {len(clean_background_files)} clean_background images.")

    for idx, image_path in enumerate(clean_background_files, 1):
        print(f"\nProcessing {idx}/{len(clean_background_files)}: {image_path}")
        try:
            process_single_image(image_path, model, device, args)
            print(f"Successfully processed: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
