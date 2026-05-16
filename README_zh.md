<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge">
  </a>
</p>

# **2D 单图 → 3D 物体生成流水线**

## *Prompt-Inpaint × AnySplat × SAM-3D-Objects 集成*

> 本仓库最初 fork 自 [xyys2003/sam3d_gs](https://github.com/xyys2003/sam3d_gs)。

------

## **摘要**

本仓库将三个开源系统串联进单条流水线，使用一条命令即可完成单图 → 多物体 3D 资产的生成：

- **Prompt-Inpaint**：基于 SAM3 的文本提示多物体分割 + 背景补全，产出有每个物体的 mask 与 clean background。
- **AnySplat**：单图前馈式 3D Gaussian Splatting 重建；额外的 RANSAC 桌面对齐将场景对齐到坐标系原点。
- **SAM-3D-Objects**：以 RGB + mask 为输入，重建单物体的 mesh 与 Gaussian。

三者通过 `pipeline/` 下的脚本以及一个由 `uv` 管理的单一虚拟环境串联起来，整条流水线由一个 shell 命令驱动。

------

# **1. 仓库结构**

```
.
├── run_object_generation_pipeline.sh   # 主入口：单图 → 3D 资产
├── pipeline/
│   ├── background_reconstruction.py       # AnySplat + 桌面 RANSAC 对齐
│   ├── objects_generation.py           # SAM-3D-Objects 多物体重建
│   ├── mesh2mjcf.py                       # 可选：把单物体 .obj 转成 MuJoCo MJCF
│   └── utils.py                           # 渲染 / IO 公共工具
└── submodule/
    ├── Prompt-Inpaint/                    # SAM3 分割 + 背景补全
    ├── AnySplat/                          # 单图 3DGS 重建
    └── Sam-3d-objects/                    # 单物体 mesh / GS 重建
```

------

# **2. 环境安装**

整个项目运行在单个由 `uv` 管理的虚拟环境 `.venv/` 中。下面的步骤面向 RTX 50 系 GPU（CUDA 12.8，PyTorch 2.7），同样在 3090 / 4090 上验证通过。

> **硬件**：推荐使用 **显存 ≥ 24 GB** 的 NVIDIA GPU。流水线会依次加载 SAM3、AnySplat、SAM-3D-Objects，其中 SAM-3D-Objects 阶段对显存最敏感。

## **2.1 克隆仓库（含子模块）**

```bash
git clone --recursive https://github.com/Yuchi-Zhang-00/sam3d_gs.git
cd sam3d_gs
```

如果克隆时忘了 `--recursive`：

```bash
git submodule update --init --recursive
```

## **2.2 安装 Python 环境**

推荐使用一键安装脚本：

```bash
bash scripts/install_env.sh
```

脚本会创建 `.venv`、安装 CUDA 12.8 版 PyTorch、子模块依赖以及项目级运行时依赖。

如果想手动一步步执行，请查阅 [`install.md`](install.md)。该文档同时记录了 SAM-3D-Objects 的几处 requirements 文件 patch 和编译 AnySplat CUDA RoPE2D 内核所需的 `kernels.cu` 修改。

## **2.3 HuggingFace 权限申请**

流水线依赖以下三个 HuggingFace 模型：

| 模型 | 使用方 | 访问 |
| --- | --- | --- |
| [`facebook/sam3`](https://huggingface.co/facebook/sam3) | Prompt-Inpaint（Stage 1） | **gated**，需在模型页面申请权限 |
| [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) | SAM-3D-Objects（Stage 3） | **gated**，需在模型页面申请权限 |
| [`lhjiang/anysplat`](https://huggingface.co/lhjiang/anysplat) | AnySplat（Stage 2） | 公开（MIT） |

在两个 gated 模型页面接受协议后，登录一次：

```bash
hf auth login
```

两个 gated 模型需要显式放置到本地，由一个 bootstrap 脚本一次性处理（登录后
跑一次即可）：

```bash
bash scripts/download_checkpoints.sh
```

| 模型 | 落地位置 |
| --- | --- |
| `facebook/sam-3d-objects` | `submodule/Sam-3d-objects/checkpoints/hf/`（Hydra 配置树，不会被 `from_pretrained` 拉取） |
| `facebook/sam3` | `submodule/Prompt-Inpaint/checkpoints/sam3.pt`（约 3.3 GB；放到本地以免 `~/.cache` 清理后丢失） |

该脚本是幂等的，且 `run_object_generation_pipeline.sh` 在首次运行时也会
自动调用它。可以通过 `--skip-sam3d`、`--skip-sam3` 或 `--force` 单独控制每
一个 stage。

`lhjiang/anysplat` 也由同一个 bootstrap 脚本拉取（落到标准的 HuggingFace
hub 缓存 `~/.cache/huggingface/hub/` 下）。它是公开模型（MIT），**不需要
`hf auth login`**；提前拉只是避免 Stage 2 首次运行时做几 GB 的下载。
传 `--skip-anysplat` 可以跳过这一步、让 AnySplat 首次运行时再 lazy 下载。

------

## **2.4 Docker 镜像（2.1–2.3 的替代方案）**

仓库提供了一份预构建镜像，包含完整环境（CUDA 12.8 基础镜像、uv 管理的
`.venv`、编译好的 AnySplat curope CUDA 扩展、所有 PyPI 依赖），已发布到
阿里云容器镜像服务：

```
crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1
crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:latest
```

用镜像可以完全跳过 §2.2；但宿主机仍然需要克隆本仓库（用于
`run_docker.sh` 启动脚本和 checkpoint 的 bind-mount 目录），以及完成
§2.3 的 HuggingFace 权限申请。

### **前置条件**

- 已安装 Docker 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；
  显存 ≥ 24 GB 的 NVIDIA GPU
- 宿主机上已经 clone 了本仓库（`git clone --recursive ...`，见 §2.1）——
  用作 `run_docker.sh` 启动脚本所在位置,以及 checkpoint / 数据 / 输出的
  bind-mount 根目录
- 完成 §2.3 的一次性 HuggingFace 设置，并在宿主机执行过
  `bash scripts/download_checkpoints.sh`。Checkpoint 留在宿主机、通过
  bind-mount 进容器，所以只需要下载一次。

### **拉取镜像**

```bash
docker pull crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1
docker tag  crpi-3nfi31esiwp28zns.cn-hangzhou.personal.cr.aliyuncs.com/open_projects_yuchi/sam3d_gs:v0.1 sam3d-gs:latest
```

`docker tag` 这一步可选。`run_docker.sh` 默认使用 `sam3d-gs:latest`；
如果不想重 tag，可以在启动时加前缀
`SAM3D_IMAGE=crpi-.../sam3d_gs:v0.1`。

### **启动容器**

```bash
./run_docker.sh                                       # 全默认（推荐）
./run_docker.sh /path/to/sam3d_gs                     # 显式传项目目录
./run_docker.sh /path/to/sam3d_gs /mnt/hf_cache       # 自定义 HF 缓存根
SAM3D_IMAGE=sam3d-gs:v0.1 ./run_docker.sh             # 指定镜像 tag
TORCH_HOME=/mnt/torch_cache ./run_docker.sh           # 自定义 torch hub 缓存
```

启动脚本会把宿主机的关键路径 bind-mount 进容器：

| 宿主机路径 | 容器路径 | 用途 |
| --- | --- | --- |
| `<repo>/submodule/Sam-3d-objects/checkpoints` | 同名 | SAM-3D-Objects 权重（gated） |
| `<repo>/submodule/Prompt-Inpaint/checkpoints` | 同名 | SAM3 权重（gated） |
| `${HF_HOME:-$HOME/.cache/huggingface}` | `/root/.cache/huggingface` | AnySplat + 其它 HF 下载 |
| `${TORCH_HOME:-$HOME/.cache/torch}` | `/root/.cache/torch` | `torch.hub` 缓存（DINOv2 等） |
| `<repo>/data` | `/opt/sam3d_gs/data` | 输入 / 输出工作目录 |
| `<repo>/example` | `/opt/sam3d_gs/example` | 自带示例输入 / 输出 |

流水线的产物会写到你指定的 scene 目录里。因为 `data/` 和 `example/`
都是 bind-mount，容器退出后这些产物会留在宿主机上。

### **在容器内运行流水线**

进入容器后你会落到 `/opt/sam3d_gs/`。镜像里 `PATH` 和 `PYTHONPATH`
已经指向自带的 `.venv`，可以直接调用 `python` 和脚本，**不需要
`source .venv/bin/activate`**。

```bash
# 自带示例：
bash run_object_generation_pipeline.sh example/example.png

# 自己的图：
bash run_object_generation_pipeline.sh data/my_scene/input_image.png
```

Stage 1 / 2 / 3 的行为和下面 §3–§4 完全一致。

### **镜像里包含什么**

- CUDA 12.8 devel 基础镜像 + Python 3.11 `.venv`,所有 PyPI 依赖
- 已编译好的 AnySplat `curope` CUDA 扩展（sm_80 / 90 / 100 / 120）
- `coacd`、`trimesh`、`mujoco`(`pipeline/mesh2mjcf.py` 开箱可用)
- 一个 `sitecustomize.py`，monkey-patch `torch.hub`，使其在本地缓存
  存在时跳过 github 的 branch ping —— 这样网络不稳时也不会再触发
  `RemoteDisconnected`(前提是 `~/.cache/torch/hub` 已有相应模型)
- 全局的 `git insteadOf` 规则，把 `https://github.com/` 重写到
  `https://gh-proxy.com/https://github.com/`，让容器内的
  `git clone` 在 github 不稳的网络上也能工作

### **镜像里不包含什么**

- 三套模型 checkpoint（SAM3 / SAM-3D-Objects / AnySplat）。它们留在
  宿主机上、通过上面的 bind-mount 进容器。在宿主机执行一次
  `scripts/download_checkpoints.sh` 即可。
- 你自己的输入数据。放到 `<repo>/data/<scene_name>/` 下，容器里通过
  `data/<scene_name>/input_image.png` 引用。

### **使用须知**

- **流水线写出的文件在宿主机上属主是 `root`**。容器内是 root 用户跑的，
  所以写进 bind-mount 目录(`data/`、`example/`、checkpoint 目录等)
  的文件，在宿主机上看到的所有者是 uid 0。两种处理方式：

  ```bash
  # 容器退出后，在宿主机改回当前用户：
  sudo chown -R $(id -u):$(id -g) data/ example/

  # 或者从一开始就让容器用宿主机的 uid 跑。
  # 优点是不用 chown,缺点是 Sam-3d-objects 里某些 EGL / pyrender
  # 代码路径在非 root 下可能跑不通,所以一般建议用上面的 chown 方案。
  # (想试的话: 编辑 run_docker.sh,给 docker run 加上
  # `--user $(id -u):$(id -g)`)
  ```

- **`gh-proxy.com` 这个重写是给国内用户准备的**。镜像里烤了一条
  `git config --global url.<proxy>.insteadOf https://github.com/` 规则,
  让容器里 `git clone` github 仓库在 GFW 网络下也能成功。**在境外网络
  环境下这个跳转是多余的,可能反而拖慢速度**。每次进容器后执行一次即可
  禁用:

  ```bash
  git config --global --unset url."https://gh-proxy.com/https://github.com/".insteadOf
  ```

  (或者自己 commit 一个去掉这条规则的镜像变体,免得每次都跑。)

------

# **3. 快速开始**

> 如果你用的是 Docker 镜像（§2.4），先跑 `./run_docker.sh` 进容器；
> 本节后面所有命令都在**容器内**原样执行。

先用仓库自带的示例图跑一遍即可（入口脚本会自动 `source .venv`，无需手动激活环境）：

```bash
bash run_object_generation_pipeline.sh example/example.png
```

默认所有产物会写到输入图像所在目录（此例中即 `example/`）。若想显式指定输出目录，可以传第二个参数：

```bash
bash run_object_generation_pipeline.sh example/example.png path/to/scene_dir
```

脚本会在同一个 `.venv` 中按顺序执行三个 stage：

1. `submodule/Prompt-Inpaint/main.py` — 分割 + 背景补全
2. `pipeline/background_reconstruction.py` — AnySplat 重建 + 桌面对齐
3. `pipeline/objects_generation.py` — 单物体 mesh / Gaussian 导出

------

# **4. 各 Stage 详解**

## **Stage 1 — Prompt-Inpaint（SAM3 分割 + 背景补全）**

```bash
python submodule/Prompt-Inpaint/main.py \
    --resize-output \
    --save-individual-masks \
    --config submodule/Prompt-Inpaint/configs/items.yml \
    --image path/to/input_image.png \
    --output-dir path/to/scene_dir
```

输出（位于 `scene_dir/`）：

- `input_image.png` — 输入图像的 resize 副本
- `clean_background.png` — 去除所有前景物体后的补全背景
- `bg_mask.png` — 用于平面拟合的桌面 mask
- `masks/<物体名>.png` — 每个物体的二值 mask

## **Stage 2 — AnySplat + 桌面对齐 3DGS**

```bash
python pipeline/background_reconstruction.py path/to/scene_dir
```

行为：

- 递归读取输入目录下每个场景文件夹中的 `clean_background.png` 和配套的 `input_image.png`。
- 运行 AnySplat 恢复相机内外参、深度、3DGS 重建结果。
- 对 `bg_mask.png` 做 RANSAC 平面拟合，结合内部 PCA 得到 OBB，构建 world → table 变换。
- 输出 Mujoco 坐标系下的对齐点云。

常用参数：

- `--model-id lhjiang/anysplat` — 覆盖 AnySplat 的 HuggingFace 模型 id
- `--align-table` / `--no-align-table` — 是否启用 RANSAC 桌面对齐并导出 `bg_aligned.ply`（默认启用）。关闭时只导出原始 `bg.ply`
- `--x-offset`、`--z-offset` — 对齐后可选的放置偏移（米）。默认 0，对齐后的点云落在原点

输出（位于 `scene_dir/`）：

- `extrinsic.npy`、`intrinsic.npy` — 相机参数（world-to-camera；像素单位内参）
- `depth.npy`、`depth_visual.png` — 来自 splat 重建的深度
- `depth_ori.npy`、`depth_ori_visual.png` — 来自原始（未补全）图像的深度
- `scale.npy` — 场景级缩放因子
- `3d_assets/bg.ply` — AnySplat 输出的原始 3DGS 场景
- `3d_assets/bg_aligned.ply` — 桌面对齐后的 3DGS 场景（仅当 `--align-table` 启用时输出，默认启用）

## **Stage 3 — SAM-3D-Objects 单物体重建**

```bash
python pipeline/objects_generation.py --input-dir path/to/scene_dir
```

常用参数：

- `--project-root submodule/Sam-3d-objects` — checkpoint 根目录
- `--tag hf` — checkpoint 子目录（`submodule/Sam-3d-objects/checkpoints/<tag>/pipeline.yaml`）
- `--seed 42`、`--save-pt`、`--save-intermediate`

针对每一个 mask，该 stage 运行 SAM-3D-Objects 推理，通过对比投影面积与平均深度恢复物体局部尺寸，并把资产以原点姿态导出。

输出（位于 `scene_dir/3d_assets/`）：

- `<物体名>.obj` — Mujoco 单位的物体 mesh
- `<物体名>.ply` — Mujoco 单位的物体 3D Gaussian
- `<物体名>_keyframe.npy` — 最终 mesh 的平均 XYZ
- 当传入 `--save-intermediate` 时，额外导出调试用的渲染和带姿态的中间产物

------

# **5. 可选工具**

## **`pipeline/mesh2mjcf.py` — mesh → MuJoCo MJCF 转换器**

一个独立的命令行工具，把单个 `.obj` 或 `.stl` 文件转成 MuJoCo MJCF 资产
（`<asset>_dependencies.xml` + `<asset>.xml` 两个 XML，以及一个 per-asset 的
mesh / texture 目录）。它**没有**被串进
`run_object_generation_pipeline.sh`；当 Stage 3 产出
`<scene>/3d_assets/<obj>.obj` 之后按需调用即可。

默认输出根目录是输入 mesh 的父目录，所以对
`scene_dir/3d_assets/cup.obj` 运行后会在输入旁边生成一个 per-asset 目录：

```
scene_dir/3d_assets/
  cup.obj                      （原输入，不变）
  cup/                         （以 obj 名命名的 per-asset 输出目录）
    cup.obj                    （输入的拷贝）
    cup.mtl                    （若多材质）
    <纹理文件>                  （MTL 引用的贴图）
    part_0.obj part_1.obj ...  （若 -cd）
    mjcf/
      cup.xml
      cup_dependencies.xml
```

emitted XML 中的 mesh 路径写作 `<asset>/<file>`，所以消费方的 MuJoCo
scene 需要把 `meshdir`（和 `texturedir`）设为输出根目录。通过
`-o/--output <dir>` 可以重定向。

### 所需依赖

走 `scripts/install_env.sh` 装环境的话，`coacd`、`trimesh`、`mujoco` 三个包
默认就装好了。下表只在你跳过一键脚本、想手动按需装时作为参考：

| 功能 | 依赖库 | 手动安装命令 |
| --- | --- | --- |
| 多材质 OBJ 自动拆分（当存在 MTL 文件时触发） | `trimesh` | `uv pip install trimesh` |
| 凸分解（`-cd`） | `coacd`、`trimesh` | `uv pip install coacd trimesh` |
| 预览查看器（`--verbose`） | `mujoco` | `uv pip install mujoco` |

### 用法

```bash
# 基本用法（使用默认颜色 / 质量 / 惯性）
python pipeline/mesh2mjcf.py path/to/cup.obj

# 自定义 RGBA、质量、对角惯性
python pipeline/mesh2mjcf.py path/to/cup.obj \
    --rgba 0.8 0.2 0.2 1.0 --mass 0.5 --diaginertia 0.01 0.01 0.005

# 自由关节 + 凸分解，得到更精确的碰撞几何
python pipeline/mesh2mjcf.py path/to/cup.obj --free_joint -cd

# 在 mujoco.viewer 中预览
python pipeline/mesh2mjcf.py path/to/cup.obj --verbose

# 一键批量转换某个场景下所有物体
for obj in scene_dir/3d_assets/*.obj; do
    python pipeline/mesh2mjcf.py "$obj" -cd
done
```

------

# **6. 常见问题**

**Q：HuggingFace 下载报 "Consistency check failed: file should be XXXX but has size YYYY"。**

HuggingFace 缓存中的 shard 损坏。清理后重试：

```bash
rm -rf submodule/Sam-3d-objects/checkpoints/hf
rm -rf ~/.cache/huggingface/hub   # 可选，更激进
bash run_object_generation_pipeline.sh path/to/input_image.png
```

也可以在调用 HuggingFace API 时通过 `force_download=True` 强制重新下载。

**Q：AnySplat 提示 "cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead"。**

CUDA 扩展没编译。请按 [`install.md`](install.md) 里的说明修改 `kernels.cu`，再执行 `python setup.py build_ext --inplace`。

**Q：Stage 1 (Prompt-Inpaint / iopaint) 报 `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`。**

`huggingface_hub` ≥ 0.26 把 `cached_download` 删掉了，但 `iopaint` 依赖的 `diffusers` 0.27.x 还在 import 它。把 `huggingface_hub` 锁到 0.25.2：

```bash
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --force-reinstall --no-deps \
    "huggingface_hub==0.25.2"
```

新走 `scripts/install_env.sh` 的环境已经带上这个 pin。

**Q：Stage 1 报 `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`。**

同一根问题的另一侧：`transformers` 5.x 会 import `huggingface_hub.is_offline_mode`，而 0.25.2 没有这个符号。把 transformers 锁到 4.48.3：

```bash
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --force-reinstall --no-deps \
    "transformers==4.48.3"
```

新走 `scripts/install_env.sh` 的环境已经带上这个 pin。

------

# **引用**

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

# **致谢**

本项目基于并整合了以下工作：

- **SAM3** — [GitHub](https://github.com/facebookresearch/sam3) · [HuggingFace](https://huggingface.co/facebook/sam3)
- **SAM-3D-Objects** — [GitHub](https://github.com/facebookresearch/sam3d) · [HuggingFace](https://huggingface.co/facebook/sam-3d-objects)
- **AnySplat** — [HuggingFace](https://huggingface.co/lhjiang/anysplat)
- **Prompt-Inpaint** — [GitHub](https://github.com/MrZoyo/Prompt-Inpaint)

感谢原作者开放其研究成果与代码。
