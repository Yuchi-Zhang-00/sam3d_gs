#!/usr/bin/env bash
set -e

############################################
# 0. Resolve project root (directory of this script)
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

############################################
# 1. Global config (paths are relative to SCRIPT_DIR)
############################################
export HF_ENDPOINT="https://hf-mirror.com"

export HF_HOME="${SCRIPT_DIR}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"

# Path to conda initialization script (usually absolute)
CONDA_SH="/data/yufei/miniconda3/etc/profile.d/conda.sh"

# Conda env names
VLLM_ENV="vllm_test"
SAM3_ENV="sam3"

# vLLM model directory (where Qwen3-VL-8B-Thinking will be downloaded)
VLLM_MODEL_DIR="${SCRIPT_DIR}/models/qwen3_vl_8b_thinking"

# Model name exposed by vLLM and used by the Python script (--llm-model-id)
SERVED_MODEL_NAME="qwen3-vl-8b-thinking"

# vLLM server port
VLLM_PORT=8001

# SAM3 agent script (Python entry)
AGENT_SCRIPT="${SCRIPT_DIR}/pipeline/run_sam3_agent_full.py"

# Input image
IMAGE_PATH="${SCRIPT_DIR}/assets/img.jpg"

# Output root directory
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/master_with_vllm"

# System prompt file for Qwen
SYSTEM_PROMPT_PATH="${SCRIPT_DIR}/assets/system_prompt_scene_prompts.txt"

# vLLM log
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_server.log"

############################################
# 2. Initialize conda
############################################
if [ -f "${CONDA_SH}" ]; then
    # Enable `conda activate`
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
else
    echo "ERROR: conda.sh not found at ${CONDA_SH}"
    exit 1
fi

############################################
# 3. HuggingFace login (interactive, in vLLM env)
############################################
echo ">>> Activating conda env: ${VLLM_ENV}"
conda activate "${VLLM_ENV}"

echo ">>> Running 'hf auth login' (you may be prompted for a token)..."
hf auth login
echo ">>> HuggingFace login finished ✓"

############################################
# 4. Download Qwen3-VL-8B-Thinking if model dir is empty
############################################
if [ ! -d "${VLLM_MODEL_DIR}" ] || [ -z "$(ls -A "${VLLM_MODEL_DIR}" 2>/dev/null)" ]; then
    echo ">>> Model directory is empty: ${VLLM_MODEL_DIR}"
    echo ">>> Auto-downloading Qwen/Qwen3-VL-8B-Thinking ..."

    mkdir -p "${VLLM_MODEL_DIR}"

    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download \
            Qwen/Qwen3-VL-8B-Thinking \
            --local-dir "${VLLM_MODEL_DIR}" \
            --local-dir-use-symlinks False
    elif command -v hf >/dev/null 2>&1; then
        hf snapshot download Qwen/Qwen3-VL-8B-Thinking \
            --local-dir "${VLLM_MODEL_DIR}" \
            --local-dir-use-symlinks False
    else
        echo "ERROR: Neither 'huggingface-cli' nor 'hf' CLI is installed."
        echo "Please install with:  pip install -U huggingface_hub"
        exit 1
    fi

    echo ">>> Model download complete!"
else
    echo ">>> Model already exists at ${VLLM_MODEL_DIR}, skip download."
fi

############################################
# 5. Start vLLM server (still in vLLM env)
############################################
echo ">>> Starting vLLM server on GPUs 6,7 ..."
CUDA_VISIBLE_DEVICES=6,7 \
vllm serve "${VLLM_MODEL_DIR}" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 65536 \
    --max-num-seqs 4 \
    --port 8001 \
    --allowed-local-media-path / \
    --served-model-name "${SERVED_MODEL_NAME}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo ">>> vLLM server started. PID = ${VLLM_PID}"
echo ">>> Logs: ${VLLM_LOG}"

echo ">>> Waiting for vLLM server to become ready..."
until curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; do
    echo "vLLM not ready yet, waiting 2s..."
    sleep 2
done
echo ">>> vLLM server is ready!"

############################################
# 6. Run SAM3 agent (in sam3 env)
############################################
echo ">>> Activating SAM3 env: ${SAM3_ENV}"
conda activate "${SAM3_ENV}"

echo ">>> Running SAM3 agent with CUDA_VISIBLE_DEVICES=0 ..."
CUDA_VISIBLE_DEVICES=0 \
python "${AGENT_SCRIPT}" \
    --image-path "${IMAGE_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --system-prompt-path "${SYSTEM_PROMPT_PATH}" \
    --llm-model-id "${SERVED_MODEL_NAME}" \
    --skip-first

echo ">>> SAM3 agent finished."

############################################
# 7. Done (vLLM is still running)
############################################
echo ">>> All done. vLLM is still running with PID = ${VLLM_PID}"
echo ">>> To stop it manually, run:  kill ${VLLM_PID}"
