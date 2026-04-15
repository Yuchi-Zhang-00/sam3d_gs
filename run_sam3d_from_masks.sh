#!/usr/bin/env bash
# 不要开 -u，会和 conda activate 脚本打架
set -eo pipefail

############################################
# 0. Resolve project root (directory of this script)
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 防止 conda activate 的 binutils 脚本里引用未定义 ADDR2LINE
export ADDR2LINE=addr2line

############################################
# 1. Global config (all paths relative to SCRIPT_DIR)
############################################

# GPU used for SAM3D reconstruction
export CUDA_VISIBLE_DEVICES="0"

# HF / Torch cache (和 run_agent_with_vllm.sh 共用一套)
export HF_ENDPOINT="https://hf-mirror.com"

# 临时取消镜像源（仅当前终端生效）
unset HF_ENDPOINT
# export HF_HOME="${SCRIPT_DIR}/huggingface"
# export TRANSFORMERS_CACHE="${HF_HOME}"
# export HF_DATASETS_CACHE="${HF_HOME}"
# export HF_HUB_CACHE="${HF_HOME}"
# export HF_HUB_ENABLE_HF_TRANSFER=0

export TORCH_HOME="${SCRIPT_DIR}/torch_hub"
export TORCH_HUB="${SCRIPT_DIR}/torch_hub"

# # Conda init script (absolute)
# CONDA_SH="/your_path/miniconda3/etc/profile.d/conda.sh"

# # Conda env for SAM3D
# SAM3D_ENV="sam3d-objects"

# sam-3d-objects repo root
PROJECT_ROOT="${SCRIPT_DIR}/Sam-3d-objects"

# Where sam-3d-objects stores intermediate .pt
PT_SAVE_DIR="${PROJECT_ROOT}/outputs/torch_save_pt"

# Checkpoints / config paths
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"
PIPELINE_YAML="${CHECKPOINTS_DIR}/hf/pipeline.yaml"

# Python entry scripts (放在 sam3d_gs/pipeline 下)
SAM3D_MULTI_SCRIPT="${SCRIPT_DIR}/pipeline/run_sam3d_multi.py"
RECONSTRUCT_SCRIPT="${SCRIPT_DIR}/pipeline/reconstruct_from_pt.py"
# FULL_PROCESS_SCRIPT="${SCRIPT_DIR}/pipeline/process.py"
FULL_PROCESS_SCRIPT="${SCRIPT_DIR}/pipeline/process_batch_new.py"

# Input image: 使用和 SAM3 agent 一样的图
# IMAGE_PATH="${SCRIPT_DIR}/assets/img.jpg"
IMAGE_PATH="${SCRIPT_DIR}/doors/11/input_image.png"
# IMAGE_PATH="${SCRIPT_DIR}/data/new-desk/input_image.jpg"

# 🔴 关键：mask-root = SAM3 agent 的 mask 输出目录
# 如果你的 run_sam3_agent_full.py 把 mask 写在：
#   outputs/master_with_vllm/masks
# 就用这一行：
MASK_ROOT="${SCRIPT_DIR}/masks/new-desk/"
# 如果暂时还用旧目录，比如 sam3/agent_output_multi/masks，可以改成：
# MASK_ROOT="${SCRIPT_DIR}/sam3/agent_output_multi/masks"

# Run configs
TAG="hf"
SEED=42
EXPORT_GIF=1   # 1 = reconstruct 时加 --export-gif，0 = 不导出 GIF

############################################
# 2. Initialize conda / uv env
############################################
# if [ -f "${CONDA_SH}" ]; then
#     # shellcheck disable=SC1090
#     source "${CONDA_SH}"
# else
#     echo "ERROR: conda.sh not found at ${CONDA_SH}"
#     exit 1
# fi

# echo ">>> Activating conda env: ${SAM3D_ENV}"
# conda activate "${SAM3D_ENV}"

mkdir -p "${PT_SAVE_DIR}"

# Initialize uv env
# source "${SCRIPT_DIR}/sam-3d-objects/.venv/bin/activate"
source "${SCRIPT_DIR}/.venv/bin/activate"
echo "python 路径$(which python)"
############################################
# 2.5. Ensure checkpoints/${TAG}/pipeline.yaml
############################################
if [ ! -f "${PIPELINE_YAML}" ]; then
    echo ">>> pipeline.yaml not found at: ${PIPELINE_YAML}"
    echo ">>> Downloading checkpoints from facebook/sam-3d-objects ..."
    echo ">>> (确保已运行 'hf auth login' 并在网页上接受模型协议)"

    # 关闭 hf_transfer（在镜像环境下容易出奇怪错误）
    export HF_HUB_ENABLE_HF_TRANSFER=0

    # 临时下载目录（避免直接弄脏 sam-3d-objects 根目录）
    # TMP_DIR="${CHECKPOINTS_DIR}/.tmp_download_${TAG}"
    TMP_DIR="${CHECKPOINTS_DIR}/.tmp_download"
    rm -rf "${TMP_DIR}"
    mkdir -p "${TMP_DIR}"

    # 1) 把远端的 checkpoints/** 全部下载到临时目录
    if command -v hf >/dev/null 2>&1; then
        hf download \
            facebook/sam-3d-objects \
            --local-dir "${TMP_DIR}" \
            --include "checkpoints/**"  \

    else
        echo "ERROR: 'hf' CLI is not installed (required for downloading checkpoints)."
        echo "       Fix: pip install -U huggingface_hub"
        rm -rf "${TMP_DIR}"
        exit 1
    fi

    # 2) 远端结构：TMP_DIR/checkpoints/...
    #    本地目标：CHECKPOINTS_DIR/TAG/...
    mkdir -p "${CHECKPOINTS_DIR}/${TAG}"

    if [ -d "${TMP_DIR}/checkpoints" ]; then
        echo ">>> Moving downloaded checkpoints into checkpoints/$ ..."
        # 把 checkpoints/* 都移到 checkpoints/hf/
        mv "${TMP_DIR}/checkpoints/"* "${CHECKPOINTS_DIR}/${TAG}"
    else
        echo "ERROR: Expected ${TMP_DIR}/checkpoints directory, but not found."
        rm -rf "${TMP_DIR}"
        exit 1
    fi

    # 清理临时目录
    rm -rf "${TMP_DIR}"

    # echo ">>> Checkpoints downloaded → ${CHECKPOINTS_DIR}/${TAG}"
    echo ">>> Checkpoints downloaded → ${CHECKPOINTS_DIR}"
    echo ">>> Expected config at: ${PIPELINE_YAML}"
else
    echo ">>> Found existing pipeline config: ${PIPELINE_YAML}"
fi


# 确保 sam-3d-objects/notebook 在 PYTHONPATH 里，供 inference 等模块 import
export PYTHONPATH="${PROJECT_ROOT}/notebook:${PYTHONPATH:-}"

############################################
# 3. Step 1 – run SAM3D multi-object & save .pt
############################################
# echo "=== [SAM3D] Step 1: run multi-object reconstruction & save .pt ==="
# CUDA_LAUNCH_BLOCKING=1 python "${SAM3D_MULTI_SCRIPT}" \
#   --image-path "${IMAGE_PATH}" \
#   --mask-root "${MASK_ROOT}" \
#   --save-dir "${PT_SAVE_DIR}" \
#   --seed "${SEED}" \
#   --project-root "${PROJECT_ROOT}"

############################################
# 4. Step 2 – reconstruct from .pt to .ply (and optional .gif)
############################################
# echo "=== [SAM3D] Step 2: reconstruct from .pt to .ply ==="

# RECONSTRUCT_CMD=(
#   python "${RECONSTRUCT_SCRIPT}"
#   --project-root "${PROJECT_ROOT}"
#   --save-dir "${PT_SAVE_DIR}"
#   --image-path "${IMAGE_PATH}"
# )

# if [ "${EXPORT_GIF}" -eq 1 ]; then
#   RECONSTRUCT_CMD+=(--export-gif)
# fi

# "${RECONSTRUCT_CMD[@]}"

############################################
# 5. Step 1 – run SAM3D multi-object , save .pt , save transformed meshs and plys
############################################
echo "=== [SAM3D] Step *:  run SAM3D multi-object , save .pt , save transformed meshs and plys ==="
CUDA_LAUNCH_BLOCKING=1 python "${FULL_PROCESS_SCRIPT}" \
  --image-path "${IMAGE_PATH}" \
  --save-dir "${PT_SAVE_DIR}" \
  --seed "${SEED}" \
  --project-root "${PROJECT_ROOT}"

echo "✅ Pipeline finished. Check ${PROJECT_ROOT}/gaussians/multi 下的 .ply/.gif 文件"
