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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python trl/scripts/sft.py \
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
  --report_to swanlab \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine_with_min_lr \
  --run_name "${RUN_NAME}" \
  --trust_remote_code True
