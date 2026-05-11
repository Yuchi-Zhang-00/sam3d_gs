#!/usr/bin/env bash
# Bootstrap gated HuggingFace checkpoints needed by the pipeline.
#
# This script handles the two models that require explicit local placement:
#
#   1. facebook/sam-3d-objects
#      The SAM-3D-Objects codepath expects a Hydra config tree at
#        submodule/Sam-3d-objects/checkpoints/<tag>/pipeline.yaml
#      which is NOT fetched by `from_pretrained`.
#
#   2. facebook/sam3
#      Prompt-Inpaint's _resolve_checkpoint() will fall back to a HuggingFace
#      auto-download, but pulling the 3.3 GB sam3.pt into the local
#      `submodule/Prompt-Inpaint/checkpoints/` keeps the weights co-located
#      with the project and survives `~/.cache` cleanups.
#
# (lhjiang/anysplat is intentionally NOT handled here; it is pulled lazily by
# AnySplat.from_pretrained on first run.)
#
# The script is idempotent: existing target files are skipped unless --force.
#
# Usage:
#   bash scripts/download_checkpoints.sh [options]
#
# Options:
#   --tag TAG       Sub-directory under submodule/Sam-3d-objects/checkpoints/
#                   for the SAM-3D-Objects bundle. Default: hf
#   --skip-sam3d    Do not download the SAM-3D-Objects bundle.
#   --skip-sam3     Do not download the SAM3 weight (sam3.pt).
#   --force         Re-download even if the target files already exist.
#   -h, --help      Show this help.
#
# Environment overrides:
#   SAM3D_CHECKPOINT_TAG    Same as --tag
#   SAM3D_MODEL_ID          SAM-3D-Objects repo id (default: facebook/sam-3d-objects)
#   SAM3_MODEL_ID           SAM3 repo id           (default: facebook/sam3)
#   SAM3_WEIGHT_FILENAME    SAM3 weight file name  (default: sam3.pt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="${SAM3D_CHECKPOINT_TAG:-hf}"
SAM3D_MODEL_ID="${SAM3D_MODEL_ID:-facebook/sam-3d-objects}"
SAM3_MODEL_ID="${SAM3_MODEL_ID:-facebook/sam3}"
SAM3_WEIGHT_FILENAME="${SAM3_WEIGHT_FILENAME:-sam3.pt}"
SKIP_SAM3D=0
SKIP_SAM3=0
FORCE=0

usage() {
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#$//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --skip-sam3d)
            SKIP_SAM3D=1
            shift
            ;;
        --skip-sam3)
            SKIP_SAM3=1
            shift
            ;;
        --force)
            FORCE=1
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

require_hf_cli() {
    if ! command -v hf >/dev/null 2>&1; then
        cat >&2 <<'EOF'
ERROR: the 'hf' CLI is not installed.
       Fix:  pip install -U huggingface_hub
       Then make sure you've accepted the relevant model agreements on
       huggingface.co and logged in with:  hf auth login
EOF
        exit 1
    fi
}

# hf_transfer occasionally trips on mirrored networks; disable it for safety.
export HF_HUB_ENABLE_HF_TRANSFER=0


download_sam3d_objects() {
    local checkpoints_dir="${PROJECT_ROOT}/submodule/Sam-3d-objects/checkpoints"
    local target_dir="${checkpoints_dir}/${TAG}"
    local pipeline_yaml="${target_dir}/pipeline.yaml"

    if [[ -f "${pipeline_yaml}" && "${FORCE}" -eq 0 ]]; then
        echo "==> [sam-3d-objects] already present: ${pipeline_yaml}"
        return 0
    fi

    require_hf_cli
    echo "==> [sam-3d-objects] downloading ${SAM3D_MODEL_ID} into ${target_dir}"

    local tmp_dir="${checkpoints_dir}/.tmp_download_${TAG}"
    rm -rf "${tmp_dir}"
    mkdir -p "${tmp_dir}"

    # Local cleanup trap (scoped to this function via a subshell would also
    # work, but we want the trap to run on Ctrl-C too).
    trap 'rm -rf "${tmp_dir}"' EXIT

    hf download "${SAM3D_MODEL_ID}" \
        --local-dir "${tmp_dir}" \
        --include "checkpoints/**"

    if [[ ! -d "${tmp_dir}/checkpoints" ]]; then
        echo "ERROR: expected ${tmp_dir}/checkpoints after download." >&2
        exit 1
    fi

    mkdir -p "${target_dir}"
    shopt -s dotglob
    mv "${tmp_dir}/checkpoints/"* "${target_dir}/"
    shopt -u dotglob

    if [[ ! -f "${pipeline_yaml}" ]]; then
        echo "ERROR: pipeline.yaml missing after move: ${pipeline_yaml}" >&2
        exit 1
    fi

    rm -rf "${tmp_dir}"
    trap - EXIT

    echo "==> [sam-3d-objects] done: ${target_dir}"
}


download_sam3() {
    local target_dir="${PROJECT_ROOT}/submodule/Prompt-Inpaint/checkpoints"
    local target_file="${target_dir}/${SAM3_WEIGHT_FILENAME}"

    if [[ -f "${target_file}" && "${FORCE}" -eq 0 ]]; then
        echo "==> [sam3] already present: ${target_file}"
        return 0
    fi

    require_hf_cli
    echo "==> [sam3] downloading ${SAM3_MODEL_ID}/${SAM3_WEIGHT_FILENAME} into ${target_dir}"

    mkdir -p "${target_dir}"
    hf download "${SAM3_MODEL_ID}" "${SAM3_WEIGHT_FILENAME}" \
        --local-dir "${target_dir}"

    if [[ ! -f "${target_file}" ]]; then
        echo "ERROR: ${target_file} missing after download." >&2
        exit 1
    fi

    echo "==> [sam3] done: ${target_file}"
}


if [[ "${SKIP_SAM3D}" -eq 0 ]]; then
    download_sam3d_objects
else
    echo "==> [sam-3d-objects] skipped (--skip-sam3d)"
fi

if [[ "${SKIP_SAM3}" -eq 0 ]]; then
    download_sam3
else
    echo "==> [sam3] skipped (--skip-sam3)"
fi

echo "==> All requested checkpoints are in place."
