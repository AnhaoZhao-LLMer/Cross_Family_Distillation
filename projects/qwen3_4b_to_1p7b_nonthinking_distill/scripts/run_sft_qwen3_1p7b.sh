#!/bin/bash
set -euo pipefail

REPO_ROOT="/code/on_policy_distillation/trl"
PROJECT_ROOT="/code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill"
CONFIG_PATH="${PROJECT_ROOT}/configs/sft_dataset.yaml"
FILTERED_DATA_PATH="${PROJECT_ROOT}/data/filtered_correct_messages_10k.jsonl"

STUDENT_MODEL_PATH="/code/models/Qwen3-1.7B-Base/"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="qwen3_4b_to_1p7b_nonthinking_sft_${RUN_TAG}"
OUTPUT_DIR="experiments/cross_family_distillation/qwen3_4b_to_1p7b_nonthinking_sft/${RUN_TAG}/checkpoints"

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Missing dataset config: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${FILTERED_DATA_PATH}" ]]; then
  echo "[ERROR] Missing filtered dataset: ${FILTERED_DATA_PATH}"
  echo "[ERROR] Run scripts/run_generate_and_filter.sh first."
  exit 1
fi

# SwanLab settings
export SWANLAB_PROJECT="qwen3_4b_to_1p7b_nonthinking_sft"
export SWANLAB_GROUP_BY="prefix"
# export SWANLAB_MODE="disabled"

# vLLM eval settings (minimal defaults; override via env if needed)
EVAL_STEPS="${EVAL_STEPS:-50}"
VLLM_EVAL_ENABLED="${VLLM_EVAL_ENABLED:-true}"
VLLM_EVAL_TEST_NAMES="${VLLM_EVAL_TEST_NAMES:-deepscaler}"
VLLM_EVAL_TEST_PATHS="${VLLM_EVAL_TEST_PATHS:-/code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/data/deepscaler/train.jsonl}"
VLLM_EVAL_TEST_SAMPLES="${VLLM_EVAL_TEST_SAMPLES:-128}"
VLLM_EVAL_GPU="${VLLM_EVAL_GPU:-1}"
VLLM_EVAL_TEMPERATURE="${VLLM_EVAL_TEMPERATURE:-0.0}"
VLLM_EVAL_MAX_NEW_TOKENS="${VLLM_EVAL_MAX_NEW_TOKENS:-1024}"
VLLM_EVAL_TP_SIZE="${VLLM_EVAL_TP_SIZE:-1}"
VLLM_EVAL_DTYPE="${VLLM_EVAL_DTYPE:-auto}"
VLLM_EVAL_MAX_MODEL_LEN="${VLLM_EVAL_MAX_MODEL_LEN:-8192}"
VLLM_EVAL_GPU_MEM_UTIL="${VLLM_EVAL_GPU_MEM_UTIL:-0.9}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${PROJECT_ROOT}/scripts/sft_with_vllm_eval.py" \
  --config "${CONFIG_PATH}" \
  --model_name_or_path "${STUDENT_MODEL_PATH}" \
  --dtype auto \
  --dataset_train_split train \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_length 5000 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${EVAL_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --report_to swanlab \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --run_name "${RUN_NAME}" \
  --trust_remote_code True \
  --vllm_eval_enabled "${VLLM_EVAL_ENABLED}" \
  --vllm_eval_test_names "${VLLM_EVAL_TEST_NAMES}" \
  --vllm_eval_test_paths "${VLLM_EVAL_TEST_PATHS}" \
  --vllm_eval_test_samples "${VLLM_EVAL_TEST_SAMPLES}" \
  --vllm_eval_gpu "${VLLM_EVAL_GPU}" \
  --vllm_eval_temperature "${VLLM_EVAL_TEMPERATURE}" \
  --vllm_eval_max_new_tokens "${VLLM_EVAL_MAX_NEW_TOKENS}" \
  --vllm_eval_tensor_parallel_size "${VLLM_EVAL_TP_SIZE}" \
  --vllm_eval_dtype "${VLLM_EVAL_DTYPE}" \
  --vllm_eval_max_model_len "${VLLM_EVAL_MAX_MODEL_LEN}" \
  --vllm_eval_gpu_memory_utilization "${VLLM_EVAL_GPU_MEM_UTIL}"
