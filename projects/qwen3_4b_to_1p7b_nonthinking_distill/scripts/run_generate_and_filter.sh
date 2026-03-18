#!/bin/bash
set -euo pipefail

REPO_ROOT="/code/on_policy_distillation/trl"
PROJECT_ROOT="/code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill"
INPUT_JSONL="/code/on_policy_distillation/trl/deepscaler_conversation.jsonl"
TEACHER_MODEL_PATH="/code/models/Qwen3-4B/"
OUTPUT_DIR="${PROJECT_ROOT}/data"

cd "${REPO_ROOT}"
mkdir -p "${PROJECT_ROOT}/scripts" "${PROJECT_ROOT}/configs" "${PROJECT_ROOT}/data"

# If not preset by the runner, default to single GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${PROJECT_ROOT}/scripts/build_distill_dataset.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_size 10000 \
  --seed 42 \
  --max_new_tokens 5000 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --min_p 0.0 \
  --n 1 \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --tensor_parallel_size 1 \
  --dtype auto \
  --max_model_len 8192 \
  --vllm_gpu_memory_utilization 0.9 \
  --batch_size 64
