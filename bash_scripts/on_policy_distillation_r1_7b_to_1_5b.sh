#!/bin/bash
set -euo pipefail

REPO_ROOT="/code/on_policy_distillation/trl"
cd "${REPO_ROOT}"

# SwanLab settings
export SWANLAB_PROJECT="cross_family_distillation_r1_7b_to_1_5b"
export SWANLAB_GROUP_BY="prefix"
# export SWANLAB_MODE="disabled"

# Student/Teacher
# Update STUDENT_MODEL_PATH to your own 1.5B efficient variant checkpoint.
STUDENT_MODEL_PATH="/code/models/R1-Distill-Qwen-1.5B"
TEACHER_MODEL_PATH="/code/models/R1-Distill-Qwen-7B/"
DATASET_PATH="deepscaler_conversation.jsonl"

# Key experiment knobs
MAX_COMPLETION_LENGTH=1200
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="r1_7b_to_1_5b_max_len_${MAX_COMPLETION_LENGTH}_${RUN_TAG}"
OUTPUT_DIR="experiments/cross_family_distillation/r1_7b_to_1_5b/max_completion_${MAX_COMPLETION_LENGTH}/${RUN_TAG}/checkpoints"

CUDA_VISIBLE_DEVICES=0 python trl/experimental/cross_family_distillation/cross_family_distillation.py \
  --model_name_or_path "${STUDENT_MODEL_PATH}" \
  --dtype auto \
  --dataset_name "${DATASET_PATH}" \
  --dataset_train_split train \
  --bf16 True \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --temperature 1 \
  --top_p 0.95 \
  --top_k 0 \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_length 3000 \
  --lmbda 1.0 \
  --logging_steps 1 \
  --report_to swanlab \
  --project from_r1_7b_to_1.5b \
  --seed 42 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type constant \
  --trust_remote_code True \
  --teacher_model_name_or_path "${TEACHER_MODEL_PATH}" \
  --teacher_tokenizer_name_or_path "${TEACHER_MODEL_PATH}" \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 30 \
  --output_dir "${OUTPUT_DIR}" \
  --use_vllm True \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_tensor_parallel_size 1 \
  --distill_method seq_level_kd \
  --eval_test_names "gsm8k" \
  --eval_test_paths "/code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl" \
  --eval_test_samples "200" \
  --eval_temperature 0.6 \
  --eval_n 1 \
  --eval_max_new_tokens 32768 \
  --eval_steps 20 \
  --run_name "${RUN_NAME}"
