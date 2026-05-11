#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <path_img> [path]"
    echo "Example: $0 data/new-desk/input_image.png"
    exit 1
fi

path_img="$1"
if [[ $# -eq 2 ]]; then
    path="$2"
else
    path="$(dirname "${path_img}")"
fi

path_img="$(realpath "${path_img}")"
path="$(realpath "${path}")"

if [[ ! -f "${path_img}" ]]; then
    echo "Input image not found: ${path_img}"
    exit 1
fi

if [[ ! -d "${path}" ]]; then
    echo "Input directory not found: ${path}"
    exit 1
fi

source "${SCRIPT_DIR}/.venv/bin/activate"

export PYTHONPATH="${SCRIPT_DIR}/submodule/Sam-3d-objects/notebook:${SCRIPT_DIR}/submodule/Sam-3d-objects:${PYTHONPATH:-}"

echo "Python: $(which python)"
echo "Image: ${path_img}"
echo "Directory: ${path}"

# Bootstrap gated HuggingFace weights on first run.
# Both models are gated; the user must have run `hf auth login` and accepted
# the model agreements for facebook/sam-3d-objects and facebook/sam3.
SAM3D_PIPELINE_YAML="${SCRIPT_DIR}/submodule/Sam-3d-objects/checkpoints/hf/pipeline.yaml"
SAM3_WEIGHT="${SCRIPT_DIR}/submodule/Prompt-Inpaint/checkpoints/sam3.pt"
if [[ ! -f "${SAM3D_PIPELINE_YAML}" || ! -f "${SAM3_WEIGHT}" ]]; then
    echo "==> One or more gated checkpoints missing locally; running bootstrap..."
    bash "${SCRIPT_DIR}/scripts/download_checkpoints.sh"
fi

echo "==> Step 1/3: Prompt-Inpaint"
python "${SCRIPT_DIR}/submodule/Prompt-Inpaint/main.py" \
    --resize-output \
    --save-individual-masks \
    --config "${SCRIPT_DIR}/submodule/Prompt-Inpaint/configs/items.yml" \
    --image "${path_img}" \
    --output-dir "${path}"

echo "==> Step 2/3: AnySplat"
python "${SCRIPT_DIR}/pipeline/background_reconstruction.py" "${path}"

echo "==> Step 3/3: Object generation"
python "${SCRIPT_DIR}/pipeline/objects_generation.py" --input-dir "${path}"

echo "Done."
