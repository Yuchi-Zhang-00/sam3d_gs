#!/usr/bin/env bash
set -euo pipefail

############################################
# 0. Resolve script & project root
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADDR2LINE=addr2line

PROJECT_ROOT="${SCRIPT_DIR}/Sam-3d-objects"
FULL_PROCESS_SCRIPT="${SCRIPT_DIR}/pipeline/process.py"

############################################
# 1. Global config
############################################
SEED=42
PT_SAVE_DIR="${PROJECT_ROOT}/outputs/torch_save_pt"

# Debug only – comment out for normal batch
# export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${PT_SAVE_DIR}"

############################################
# 2. Environment
############################################
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
    echo "🐍 使用虚拟环境: $(which python)"
else
    echo "⚠️ 未找到 .venv，使用系统 python: $(which python)"
fi

############################################
# 3. Argument parsing
############################################
scene_name="datacol1_toykitchen1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene|-s)
            scene_name="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [--scene 场景名]"
            echo "示例: $0 --scene datacol1_toykitchen1"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            exit 1
            ;;
    esac
done

############################################
# 4. Path sanity checks
############################################
DATA_ROOT="${SCRIPT_DIR}/data/${scene_name}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "❌ PROJECT_ROOT 不存在: ${PROJECT_ROOT}"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "❌ DATA_ROOT 不存在: ${DATA_ROOT}"
    exit 1
fi

if [[ ! -f "${FULL_PROCESS_SCRIPT}" ]]; then
    echo "❌ process.py 不存在: ${FULL_PROCESS_SCRIPT}"
    exit 1
fi

############################################
# 5. Counters
############################################
PROCESSED_COUNT=0
SKIPPED_COUNT=0
ERROR_COUNT=0

echo "========================================"
echo "📂 Scene: ${scene_name}"
echo "📁 Data root: ${DATA_ROOT}"
echo "========================================"

############################################
# 6. Main loop
############################################
for SUBDIR in "${DATA_ROOT}"/*/; do
    [[ -d "${SUBDIR}" ]] || continue

    TRAJ_DIR="${SUBDIR%/}"
    TRAJ_NAME="$(basename "${TRAJ_DIR}")"
    IMAGE_PATH="${TRAJ_DIR}/input_image.png"
    IMAGE_PATH_BG="${TRAJ_DIR}/clean_background.png"
    LOG_DIR="${TRAJ_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    OUT_PT="${PT_SAVE_DIR}/${TRAJ_NAME}.pt"

    # Check image
    if [[ ! -f "${IMAGE_PATH}" ]]; then
        echo "⚠️ 跳过 ${TRAJ_NAME}: 未找到 input_image.png"
        ((SKIPPED_COUNT++))
        continue
    fi

    if [[ ! -f "${IMAGE_PATH_BG}" ]]; then
        echo "⚠️ 跳过 ${TRAJ_NAME}: 未找到 clean_background.png"
        ((SKIPPED_COUNT++))
        continue
    fi

    # Skip if already done
    if [[ -f "${OUT_PT}" ]]; then
        echo "⏭️ 跳过 ${TRAJ_NAME}: 已存在输出 ${OUT_PT}"
        ((SKIPPED_COUNT++))
        continue
    fi

    ((PROCESSED_COUNT++))
    FAILED=false

    echo ""
    echo "========================================"
    echo "📁 处理: ${TRAJ_NAME}"
    echo "🖼️  ${IMAGE_PATH}"
    echo "========================================"

    ############################################
    # Step 1 – AnySplat
    ############################################
    echo "▶ Step 1: AnySplat"
    if ! python "${PROJECT_ROOT}/AnySplat/inference_ransanc.py" \
        "${IMAGE_PATH_BG}" \
        > "${LOG_DIR}/anysplat.log" 2>&1; then
        echo "❌ AnySplat 失败 (${TRAJ_NAME})"
        FAILED=true
    fi

    ############################################
    # Step 2 – SAM3D
    ############################################
    echo "▶ Step 2: SAM3D"
    if ! python "${FULL_PROCESS_SCRIPT}" \
        --image-path "${IMAGE_PATH}" \
        --save-dir "${PT_SAVE_DIR}" \
        --seed "${SEED}" \
        --project-root "${PROJECT_ROOT}" \
        --traj-name "${TRAJ_NAME}" \
        > "${LOG_DIR}/sam3d.log" 2>&1; then
        echo "❌ SAM3D 失败 (${TRAJ_NAME})"
        FAILED=true
    fi

    ############################################
    # Result
    ############################################
    if $FAILED; then
        ((ERROR_COUNT++))
        echo "❌ ${TRAJ_NAME} 处理失败"
    else
        echo "✅ ${TRAJ_NAME} 处理完成"
    fi
done

############################################
# 7. Summary
############################################
echo ""
echo "========================================"
echo "📊 批处理结果汇总"
echo "========================================"
echo "✅ 成功处理: $((PROCESSED_COUNT - ERROR_COUNT))"
echo "⚠️  跳过: ${SKIPPED_COUNT}"
echo "❌  失败: ${ERROR_COUNT}"
echo "📁 输出目录: ${PT_SAVE_DIR}"
echo "========================================"
