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

The project runs inside a single `uv`-managed virtual environment (`.venv/`). The setup below targets RTX 50-series GPUs (CUDA 12.8, PyTorch 2.7) and is also verified to work on 3090 / 4090.

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

`lhjiang/anysplat` is also fetched by the same bootstrap script (into the
standard HuggingFace hub cache at `~/.cache/huggingface/hub/`). It is public
(MIT), so no `hf auth login` is required for this one — pre-fetching just
keeps the first Stage-2 run from doing a multi-GB download. Pass
`--skip-anysplat` if you'd rather have AnySplat pull it lazily on first run.

------

## **2.4 Docker image (alternative to 2.1–2.3)**

A pre-built image with the full environment (CUDA 12.8 base, the
uv-managed `.venv`, the compiled AnySplat curope CUDA extension, and all
PyPI deps) is published to Aliyun Container Registry:

```
crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1
crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:latest
```

Using the image skips §2.2 entirely; you still need a clone of this repo on
the host (the launcher and the host-side checkpoint directories) and HF
access for the two gated models (§2.3).

### **Prerequisites**

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed; an NVIDIA GPU with ≥ 24 GB VRAM
- A local clone of this repo (`git clone --recursive ...`, see §2.1) — used
  both for the `run_docker.sh` launcher and as the bind-mount root for
  checkpoints, data, and outputs
- One-time HuggingFace setup (§2.3) and a host-side run of
  `bash scripts/download_checkpoints.sh`. Checkpoints live on the host and
  are bind-mounted into the container, so this only runs once.

### **Pull the image**

```bash
docker pull crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1
docker tag  crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1 sam3d-gs:latest
```

The re-tag is optional. `run_docker.sh` defaults to `sam3d-gs:latest`; if
you'd rather not re-tag, prefix the launch with
`SAM3D_IMAGE=crpi-.../sam3d_gs:v0.1` instead.

### **Launch the container**

```bash
./run_docker.sh                                       # uses defaults
./run_docker.sh /path/to/sam3d_gs                     # explicit project dir
./run_docker.sh /path/to/sam3d_gs /mnt/hf_cache       # custom HF cache root
SAM3D_IMAGE=sam3d-gs:v0.1 ./run_docker.sh             # pick a specific tag
TORCH_HOME=/mnt/torch_cache ./run_docker.sh           # custom torch hub cache
```

The launcher bind-mounts the relevant host paths into the container:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `<repo>/submodule/Sam-3d-objects/checkpoints` | same | SAM-3D-Objects weights (gated) |
| `<repo>/submodule/Prompt-Inpaint/checkpoints` | same | SAM3 weight (gated) |
| `${HF_HOME:-$HOME/.cache/huggingface}` | `/root/.cache/huggingface` | AnySplat + other HF downloads |
| `${TORCH_HOME:-$HOME/.cache/torch}` | `/root/.cache/torch` | `torch.hub` cache (DINOv2 etc.) |
| `<repo>/data` | `/opt/sam3d_gs/data` | scratch input/output dir |
| `<repo>/example` | `/opt/sam3d_gs/example` | bundled demo input/output |

Pipeline outputs land in whichever scene directory you point the launcher
at — since `data/` and `example/` are bind-mounted, those outputs persist
on the host after the container exits.

### **Run the pipeline inside the container**

You land in `/opt/sam3d_gs/`. The image's `PATH` and `PYTHONPATH` already
point at the bundled `.venv`, so you can call `python` and run scripts
directly — **no `source .venv/bin/activate`**.

```bash
# Bundled demo:
bash run_object_generation_pipeline.sh example/example.png

# Your own image:
bash run_object_generation_pipeline.sh data/my_scene/input_image.png
```

Stage 1/2/3 each behave exactly as in §3–§4 below.

### **What's baked into the image**

- CUDA 12.8 devel base + Python 3.11 `.venv` with every PyPI dep
- Compiled AnySplat `curope` CUDA extension (sm_80 / 90 / 100 / 120)
- `coacd`, `trimesh`, `mujoco` (so `pipeline/mesh2mjcf.py` works out of the box)
- `sitecustomize.py` patching `torch.hub` to use the local cache without
  pinging github first (avoids `RemoteDisconnected` on flaky networks once
  the model is in `~/.cache/torch/hub`)
- A global `git insteadOf` rule routing `https://github.com/` through
  `https://gh-proxy.com/https://github.com/`, so in-container `git clone`
  works on networks where direct github access is unreliable

### **What's NOT baked in**

- The three model checkpoint sets (SAM3, SAM-3D-Objects, AnySplat). They
  live on the host and are bind-mounted via the table above. Run
  `scripts/download_checkpoints.sh` once on the host.
- Your input data. Drop it into `<repo>/data/<scene_name>/` and reference
  it as `data/<scene_name>/input_image.png` inside the container.

### **Caveats**

- **Output files end up owned by `root` on the host.** The container runs
  as root, so anything the pipeline writes into a bind-mounted directory
  (`data/`, `example/`, the checkpoint dirs, etc.) shows up on the host
  with uid 0. Two ways to deal with it:

  ```bash
  # After the container exits, fix ownership on the host:
  sudo chown -R $(id -u):$(id -g) data/ example/

  # Or run the container as your host user from the start.
  # This avoids the chown step but can break EGL / pyrender setup
  # in some Sam-3d-objects code paths, so prefer the chown fix.
  # (To try anyway: edit run_docker.sh and add `--user $(id -u):$(id -g)`
  # to the `docker run` invocation.)
  ```

- **The `gh-proxy.com` redirect is for users behind the GFW.** The image
  bakes a `git config --global url.<proxy>.insteadOf https://github.com/`
  rule so in-container `git clone` of github URLs survives flaky direct
  access from mainland China. **Outside mainland China this hop is
  unnecessary and may slow things down.** Disable it once per container
  start:

  ```bash
  git config --global --unset url."https://gh-proxy.com/https://github.com/".insteadOf
  ```

  (Or bake your own image variant with the rule removed if you'd rather
  not run that every time.)

------

# **3. Quick Start**

> If you're using the Docker image (§2.4), start the container first with
> `./run_docker.sh` — every command in this section runs **inside** the
> container exactly as written.

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

Fresh installs via `scripts/install_env.sh` already include all three optional
packages (`coacd`, `trimesh`, `mujoco`), so the table below is only for
reference if you skip the bundled installer or build the environment
piecemeal:

| Feature | Library | Manual install |
| --- | --- | --- |
| Multi-material OBJ splitting (automatic when an MTL file is present) | `trimesh` | `uv pip install trimesh` |
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

**Q: `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` during Stage 1 (Prompt-Inpaint / iopaint).**

`huggingface_hub` ≥ 0.26 removed `cached_download`, but `diffusers` 0.27.x (which is what `iopaint` pulls in) still imports it. Downgrade `huggingface_hub` to 0.25.2:

```bash
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --force-reinstall --no-deps \
    "huggingface_hub==0.25.2"
```

Fresh installs via `scripts/install_env.sh` already include this pin.

**Q: `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'` during Stage 1.**

Same symptom from the other direction: `transformers` 5.x imports `is_offline_mode` from `huggingface_hub`, which doesn't exist in 0.25.2. Pin transformers to 4.48.3:

```bash
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --force-reinstall --no-deps \
    "transformers==4.48.3"
```

Fresh installs via `scripts/install_env.sh` already include this pin.

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
