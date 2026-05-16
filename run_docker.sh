#!/usr/bin/env bash
# Launch sam3d-gs:latest with host checkpoints + data bind-mounted.
#
# Usage:
#   run_docker.sh [PROJECT_DIR] [HF_CACHE_DIR]
#
# PROJECT_DIR    Path to the sam3d_gs repo on the host.
#                Defaults to the directory this script lives in.
# HF_CACHE_DIR   Path to host HuggingFace cache (so AnySplat and other
#                HF models are reused across container starts).
#                Defaults to ${HF_HOME:-$HOME/.cache/huggingface}.
#
# Environment overrides:
#   SAM3D_IMAGE  Docker image to run.  Default: sam3d-gs:latest
#   TORCH_HOME   Host PyTorch hub cache (DINOv2 etc. land here).
#                Default: $HOME/.cache/torch

set -euo pipefail

DEFAULT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-${DEFAULT_REPO}}"
HF_CACHE="${2:-${HF_HOME:-${HOME}/.cache/huggingface}}"
TORCH_CACHE="${TORCH_HOME:-${HOME}/.cache/torch}"
IMAGE="${SAM3D_IMAGE:-sam3d-gs:latest}"

REPO="$(realpath "${REPO}")"
HF_CACHE="$(realpath -m "${HF_CACHE}")"
TORCH_CACHE="$(realpath -m "${TORCH_CACHE}")"

# Sanity-check that PROJECT_DIR really looks like the sam3d_gs repo.
for marker in submodule/Sam-3d-objects submodule/Prompt-Inpaint scripts/install_env.sh; do
    if [[ ! -e "${REPO}/${marker}" ]]; then
        echo "ERROR: ${REPO} does not look like a sam3d_gs checkout (missing ${marker})." >&2
        echo "Pass the project root explicitly: $0 /path/to/sam3d_gs" >&2
        exit 1
    fi
done

# Ensure host-side bind targets exist (Docker would otherwise create them as root).
mkdir -p \
    "${REPO}/submodule/Sam-3d-objects/checkpoints" \
    "${REPO}/submodule/Prompt-Inpaint/checkpoints" \
    "${REPO}/data" \
    "${REPO}/example" \
    "${HF_CACHE}" \
    "${TORCH_CACHE}"

echo "==> repo:        ${REPO}"
echo "==> hf cache:    ${HF_CACHE}"
echo "==> torch cache: ${TORCH_CACHE}"
echo "==> image:       ${IMAGE}"

docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    --network host \
    -v "${REPO}/submodule/Sam-3d-objects/checkpoints":/opt/sam3d_gs/submodule/Sam-3d-objects/checkpoints \
    -v "${REPO}/submodule/Prompt-Inpaint/checkpoints":/opt/sam3d_gs/submodule/Prompt-Inpaint/checkpoints \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    -v "${TORCH_CACHE}":/root/.cache/torch \
    -v "${REPO}/data":/opt/sam3d_gs/data \
    -v "${REPO}/example":/opt/sam3d_gs/example \
    "${IMAGE}"
