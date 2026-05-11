<p align="center">
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/语言-中文-red?style=for-the-badge">
  </a>
</p>

# **Unified 2D Single-Image → 3D Object Generation Pipeline**

## *Prompt-Inpaint × AnySplat × SAM-3D-Objects Integration*

> This repo was originally forked from [xyys2003/sam3d_gs](https://github.com/xyys2003/sam3d_gs).

------

## **Abstract**

This repository packages a single-image 2D → 3D object reconstruction pipeline by composing three open-source systems behind one entry script:

- **Prompt-Inpaint** — text-prompted multi-object segmentation (built on SAM3) plus background inpainting, producing per-object masks and a clean background image.
- **AnySplat** — feed-forward 3D Gaussian Splatting from a single image, plus a RANSAC-based table-alignment pass that brings the scene into a Mujoco-friendly world frame.
- **SAM-3D-Objects** — per-object mesh and Gaussian reconstruction from RGB + mask.

The three components are wired together through scripts under `pipeline/` and a single uv-managed virtual environment, so the whole pipeline runs from one shell command.

------

# **1. Repository Layout**

```
.
├── run_object_generation_pipeline.sh   # one-shot entry: image → 3D assets
├── pipeline/
│   ├── background_reconstruction.py       # AnySplat + table RANSAC alignment
│   ├── objects_generation.py           # SAM-3D-Objects multi-object reconstruction
│   ├── mesh2mjcf.py                       # optional: convert per-object .obj → MuJoCo MJCF
│   └── utils.py                           # shared rendering / IO helpers
└── submodule/
    ├── Prompt-Inpaint/                    # SAM3 segmentation + inpainting
    ├── AnySplat/                          # single-image 3DGS reconstruction
    └── Sam-3d-objects/                    # per-object mesh / GS reconstruction
```

------

# **2. Setup**

The project runs inside a single `uv`-managed virtual environment (`.venv/`). The instructions below cover an Ada / 50-series GPU build (CUDA 12.8, PyTorch 2.7).

> **Hardware**: an NVIDIA GPU with **≥ 24 GB VRAM** is recommended. The pipeline loads SAM3, AnySplat, and SAM-3D-Objects sequentially and the SAM-3D-Objects stage in particular is memory-hungry.

## **2.1 Clone with submodules**

```bash
git clone --recursive https://github.com/Yuchi-Zhang-00/sam3d_gs.git
cd sam3d_gs
```

If the submodules were not initialized at clone time:

```bash
git submodule update --init --recursive
```

## **2.2 Install the Python environment**

The recommended path is the bundled one-command installer:

```bash
bash scripts/install_env.sh
```

It creates `.venv`, installs PyTorch for CUDA 12.8, the submodule dependencies, and the project-level runtime dependencies.

If you would rather run each step yourself, see [`install.md`](install.md). It also documents the small SAM-3D-Objects requirements-file patches and the AnySplat `kernels.cu` fix used to build the CUDA RoPE2D kernel.

## **2.3 HuggingFace access**

The pipeline pulls three models from HuggingFace:

| Model | Used by | Access |
| --- | --- | --- |
| [`facebook/sam3`](https://huggingface.co/facebook/sam3) | Prompt-Inpaint (Stage 1) | **Gated** — request access on the model page |
| [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) | SAM-3D-Objects (Stage 3) | **Gated** — request access on the model page |
| [`lhjiang/anysplat`](https://huggingface.co/lhjiang/anysplat) | AnySplat (Stage 2) | Public (MIT) |

After accepting the agreements on the two gated pages, log in once:

```bash
hf auth login
```

The two gated models need explicit local placement and are fetched by a
single bootstrap script (run once, after `hf auth login`):

```bash
bash scripts/download_checkpoints.sh
```

| Model | Target |
| --- | --- |
| `facebook/sam-3d-objects` | `submodule/Sam-3d-objects/checkpoints/hf/` (Hydra config tree, not fetched by `from_pretrained`) |
| `facebook/sam3` | `submodule/Prompt-Inpaint/checkpoints/sam3.pt` (~3.3 GB; placed locally so it isn't lost when `~/.cache` is cleaned) |

The script is idempotent and is also invoked automatically by
`run_object_generation_pipeline.sh` on first run. Use `--skip-sam3d`,
`--skip-sam3`, or `--force` to control individual stages.

`lhjiang/anysplat` is public and is downloaded lazily by
`AnySplat.from_pretrained` the first time Stage 2 runs — no login or
bootstrap step is needed for it.

------

# **3. Quick Start**

Try the bundled demo image (the entry script activates `.venv` internally, so you don't need to do it yourself):

```bash
bash run_object_generation_pipeline.sh example/example.png
```

By default, all outputs are written next to the input image (in this case, into `example/`). Pass an explicit output directory as the second argument if you want them elsewhere:

```bash
bash run_object_generation_pipeline.sh example/example.png path/to/scene_dir
```

The script runs three stages in sequence inside the single `.venv`:

1. `submodule/Prompt-Inpaint/main.py` — segmentation + inpainting
2. `pipeline/background_reconstruction.py` — AnySplat reconstruction + table alignment
3. `pipeline/objects_generation.py` — per-object mesh + Gaussian export

------

# **4. Pipeline Stages**

## **Stage 1 — Prompt-Inpaint (SAM3 segmentation + inpainting)**

```bash
python submodule/Prompt-Inpaint/main.py \
    --resize-output \
    --save-individual-masks \
    --config submodule/Prompt-Inpaint/configs/items.yml \
    --image path/to/input_image.png \
    --output-dir path/to/scene_dir
```

Outputs (under `scene_dir/`):

- `input_image.png` — resized copy of the input
- `clean_background.png` — inpainted background with all foreground objects removed
- `bg_mask.png` — table / desktop mask used for plane fitting
- `masks/<object_name>.png` — per-object binary masks

## **Stage 2 — AnySplat + table-aligned 3D Gaussians**

```bash
python pipeline/background_reconstruction.py path/to/scene_dir
```

Behaviour:

- Loads `clean_background.png` (and the matching `input_image.png`) inside each scene folder under the input directory.
- Runs AnySplat to recover camera intrinsics/extrinsics, depth, and a 3DGS reconstruction.
- Fits a RANSAC plane to `bg_mask.png`, derives an OBB via inner PCA, and builds a world-to-table transform.
- Re-emits the splat in a Mujoco-friendly frame.

Useful flags:

- `--model-id lhjiang/anysplat` — override the AnySplat HuggingFace model id
- `--align-table` / `--no-align-table` — toggle RANSAC table alignment + the `bg_aligned.ply` export (default: enabled). When disabled, only the raw `bg.ply` is written
- `--x-offset`, `--z-offset` — optional placement offsets (m) applied after alignment. Default: 0, so the aligned cloud sits at the origin

Outputs (under `scene_dir/`):

- `extrinsic.npy`, `intrinsic.npy` — camera parameters (world-to-camera; pixel-unit intrinsics)
- `depth.npy`, `depth_visual.png` — depth from the splat reconstruction
- `depth_ori.npy`, `depth_ori_visual.png` — depth from the original (non-inpainted) image
- `scale.npy` — scene-level scale factor
- `3d_assets/bg.ply` — raw 3DGS scene from AnySplat
- `3d_assets/bg_aligned.ply` — table-aligned 3DGS scene (only when `--align-table` is on, which is the default)

## **Stage 3 — SAM-3D-Objects per-object reconstruction**

```bash
python pipeline/objects_generation.py --input-dir path/to/scene_dir
```

Useful flags:

- `--project-root submodule/Sam-3d-objects` — checkpoint root
- `--tag hf` — checkpoint subdirectory (`submodule/Sam-3d-objects/checkpoints/<tag>/pipeline.yaml`)
- `--seed 42`, `--save-pt`, `--save-intermediate`

For each mask, the stage runs SAM-3D-Objects inference, recovers the object's local scale by matching projected area + mean depth against the AnySplat depth map, and exports the asset at the origin.

Outputs (under `scene_dir/3d_assets/`):

- `<object>.obj` — per-object mesh sized for Mujoco
- `<object>.ply` — per-object 3D Gaussians sized for Mujoco
- `<object>_keyframe.npy` — mean XYZ of the final mesh
- (with `--save-intermediate`) debug renderings and the pose-applied versions

------

# **5. Optional Tools**

## **`pipeline/mesh2mjcf.py` — mesh → MuJoCo MJCF converter**

A standalone CLI that turns a single `.obj` or `.stl` mesh into MuJoCo MJCF
assets (a `<asset>_dependencies.xml` + `<asset>.xml` pair, plus a per-asset
mesh / texture directory). It is **not** wired into
`run_object_generation_pipeline.sh`; use it on demand once Stage 3 has
produced `<scene>/3d_assets/<obj>.obj`.

By default, the output root is the parent directory of the input mesh, so
running it on `scene_dir/3d_assets/cup.obj` writes a self-contained per-asset
folder right next to the input:

```
scene_dir/3d_assets/
  cup.obj                      (original input, untouched)
  cup/                         (per-asset output folder, named after the obj stem)
    cup.obj                    (copy of the input)
    cup.mtl                    (if multi-material)
    <texture files>            (referenced by the MTL)
    part_0.obj part_1.obj ...  (if -cd)
    mjcf/
      cup.xml
      cup_dependencies.xml
```

Mesh paths inside the emitted XMLs are written as `<asset>/<file>`, so the
consuming MuJoCo scene should set `meshdir` (and `texturedir`) to the output
root. Pass `-o/--output <dir>` to redirect.

### Required libraries

The converter has no extra dependencies beyond the Python standard library
unless you opt into the following features:

| Feature | Library | Install |
| --- | --- | --- |
| Multi-material OBJ splitting (automatic when an MTL file is present) | `trimesh` | usually already installed via the Sam-3d-objects extras; `uv pip install trimesh` otherwise |
| Convex decomposition (`-cd`) | `coacd`, `trimesh` | `uv pip install coacd trimesh` |
| Preview viewer (`--verbose`) | `mujoco` | `uv pip install mujoco` |

### Usage

```bash
# Basic conversion (default colour / mass / inertia)
python pipeline/mesh2mjcf.py path/to/cup.obj

# Custom RGBA, mass, and diagonal inertia
python pipeline/mesh2mjcf.py path/to/cup.obj \
    --rgba 0.8 0.2 0.2 1.0 --mass 0.5 --diaginertia 0.01 0.01 0.005

# Free-floating body + convex decomposition for accurate collisions
python pipeline/mesh2mjcf.py path/to/cup.obj --free_joint -cd

# Preview in mujoco.viewer after conversion
python pipeline/mesh2mjcf.py path/to/cup.obj --verbose

# Batch over all per-object meshes in one scene
for obj in scene_dir/3d_assets/*.obj; do
    python pipeline/mesh2mjcf.py "$obj" -cd
done
```

------

# **6. FAQ**

**Q: HuggingFace download fails with “Consistency check failed: file should be XXXX but has size YYYY”.**

Corrupt shards in the HuggingFace cache. Clear and retry:

```bash
rm -rf submodule/Sam-3d-objects/checkpoints/hf
rm -rf ~/.cache/huggingface/hub   # optional, more aggressive
bash run_object_generation_pipeline.sh path/to/input_image.png
```

You can also force a fresh download by setting `force_download=True` when invoking the HuggingFace API.

**Q: AnySplat reports “cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead”.**

The CUDA extension was not built. Apply the `kernels.cu` patch documented in [`install.md`](install.md) and run `python setup.py build_ext --inplace`.

------

# **Citations**

```bibtex
@article{kirillov2024sam3,
  title  = {SAM 3: Segment Anything in Images and Videos},
  author = {Kirillov, Alexander and Ravi, Nikhila and Mao, Weiyao and others},
  year   = {2024},
  url    = {https://github.com/facebookresearch/sam3}
}

@article{wu2024sam3dobjects,
  title  = {SAM-3D-Objects: Segment Anything in 3D Using 2D Masks},
  author = {Wu, Yu and Mao, Weiyao and Kirillov, Alexander and others},
  year   = {2024},
  url    = {https://github.com/facebookresearch/sam-3d-objects}
}

@article{jiang2024anysplat,
  title  = {AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views},
  author = {Jiang, Lihan and others},
  year   = {2024},
  url    = {https://github.com/OpenRobotLab/AnySplat}
}
```

------

# **Acknowledgements**

This project is built upon and integrates:

- **SAM3** — [GitHub](https://github.com/facebookresearch/sam3) · [HuggingFace](https://huggingface.co/facebook/sam3)
- **SAM-3D-Objects** — [GitHub](https://github.com/facebookresearch/sam3d) · [HuggingFace](https://huggingface.co/facebook/sam-3d-objects)
- **AnySplat** — [HuggingFace](https://huggingface.co/lhjiang/anysplat)
- **Prompt-Inpaint** — [GitHub](https://github.com/MrZoyo/Prompt-Inpaint)

We thank the authors for making their research and implementations publicly available.
