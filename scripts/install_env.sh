#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
KAOLIN_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html"

INSTALL_TORCH=1
UPDATE_SUBMODULES=1
COMPILE_CUROPE=1

usage() {
    cat <<'EOF'
Usage: bash scripts/install_env.sh [options]

Options:
  --python VERSION        Python version for uv venv. Default: 3.11
  --skip-torch           Do not install torch/torchvision/torchaudio.
  --skip-submodules      Do not run git submodule update --init --recursive.
  --skip-curope          Do NOT patch+compile AnySplat curope CUDA extension
                         (compiled by default; without it AnySplat falls back
                         to a slower PyTorch RoPE2D implementation).
  -h, --help             Show this help.

Examples:
  bash scripts/install_env.sh
  bash scripts/install_env.sh --skip-torch
  bash scripts/install_env.sh --skip-curope
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-torch)
            INSTALL_TORCH=0
            shift
            ;;
        --skip-submodules)
            UPDATE_SUBMODULES=0
            shift
            ;;
        --skip-curope)
            COMPILE_CUROPE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

echo "==> Project root: ${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found. Installing uv with pip..."
    python3 -m pip install -U uv
fi

if [[ "${UPDATE_SUBMODULES}" -eq 1 ]]; then
    echo "==> Updating git submodules..."
    git submodule update --init --recursive
fi

echo "==> Creating/updating .venv with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" .venv

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}/submodule/Sam-3d-objects/notebook:${PROJECT_ROOT}/submodule/Sam-3d-objects:${PYTHONPATH:-}"
export PIP_FIND_LINKS="${KAOLIN_FIND_LINKS}"

echo "==> Python: $(which python)"
python --version

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
    echo "==> Installing PyTorch ${TORCH_VERSION} from ${PYTORCH_INDEX_URL}..."
    uv pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}"
else
    echo "==> Skipping PyTorch install."
fi

echo "==> Installing AnySplat requirements..."
uv pip install -r submodule/AnySplat/requirements.txt --no-build-isolation

echo "==> Installing SAM-3D-Objects build helpers..."
uv pip install hatch-requirements-txt editables wheel

echo "==> Installing SAM-3D-Objects extras..."
uv pip install -e './submodule/Sam-3d-objects[dev]'
uv pip install -e './submodule/Sam-3d-objects[p3d]' --no-build-isolation
uv pip install -e './submodule/Sam-3d-objects[inference]' \
    --no-build-isolation \
    --find-links "${KAOLIN_FIND_LINKS}"

echo "==> Installing project-level runtime dependencies..."
# Do NOT use -U here: that would let uv upgrade transitive deps (notably
# torch, via iopaint) and clobber the CUDA-pinned torch installed above.
uv pip install --index-strategy unsafe-best-match \
    "transformers==4.48.3" \
    "iopaint>=1.2.0" \
    "diffusers>=0.27.2" \
    "numpy<2.0" \
    "opencv-python>=4.8.0" \
    "pyyaml>=6.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "setuptools" \
    "einops"

# Pin huggingface_hub to 0.25.2 as the very last step: diffusers 0.27.2 (and
# the iopaint stack on top of it) still imports `cached_download` from
# huggingface_hub, which was removed in hub >= 0.26. Upstream Sam-3d-objects /
# iopaint extras may pull in a newer hub transitively, so we force-reinstall
# last (with --no-deps so it can downgrade without uv complaining) and lock
# the exact version that was empirically verified to work.
#
# Note: transformers above is pinned to ==4.48.3 (not >=) because transformers
# 5.x imports `is_offline_mode` from huggingface_hub, which doesn't exist in
# 0.25.2 — using a floor here lets pip resolve to 5.x and breaks iopaint at
# runtime even though hub stays pinned.
echo "==> Pinning huggingface_hub==0.25.2 (force-reinstall, no-deps)..."
uv pip install --index-strategy unsafe-best-match --force-reinstall --no-deps \
    "huggingface_hub==0.25.2"

echo "==> Installing SAM3..."
uv pip install --index-strategy unsafe-best-match \
    "git+https://github.com/facebookresearch/sam3.git"

# Optional mesh2mjcf extras (installed by default so `-cd` / `--verbose` Just
# Work; `trimesh` is also used for multi-material OBJ splitting).
echo "==> Installing mesh2mjcf extras (coacd, trimesh, mujoco)..."
uv pip install --index-strategy unsafe-best-match \
    "coacd" \
    "trimesh" \
    "mujoco"

if [[ "${COMPILE_CUROPE}" -eq 1 ]]; then
    CUROPE_DIR="${PROJECT_ROOT}/submodule/AnySplat/src/model/encoder/backbone/croco/curope"
    KERNELS_CU="${CUROPE_DIR}/kernels.cu"

    if [[ ! -f "${KERNELS_CU}" ]]; then
        echo "ERROR: kernels.cu not found: ${KERNELS_CU}" >&2
        exit 1
    fi

    echo "==> Patching AnySplat curope kernels.cu..."
    python - "${KERNELS_CU}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
patched = text.replace(
    'AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {',
    'AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {',
)
if patched != text:
    path.write_text(patched)
    print(f"patched {path}")
else:
    print(f"no patch needed for {path}")
PY

    echo "==> Building AnySplat curope extension..."
    (
        cd "${CUROPE_DIR}"
        python setup.py build_ext --inplace
    )
fi

cat <<EOF

==> Install finished.

Next steps:
  source .venv/bin/activate
  export PYTHONPATH="${PROJECT_ROOT}/submodule/Sam-3d-objects/notebook:${PROJECT_ROOT}/submodule/Sam-3d-objects:\${PYTHONPATH:-}"

If you use gated HuggingFace models, run:
  huggingface-cli login
EOF
