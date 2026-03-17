#!/bin/bash
set -euo pipefail
REPO_ROOT="/code/on_policy_distillation/trl"
cd "${REPO_ROOT}"
# 设置 SwanLab 环境变量
export SWANLAB_PROJECT="cross_family_distillation_efficient"
export SWANLAB_GROUP_BY="prefix"
# export SWANLAB_MODE="disabled"

CUDA_VISIBLE_DEVICES=0 python trl/experimental/cross_family_distillation/cross_family_distillation.py \
  --model_name_or_path /code/models/R1-Distill-Qwen-1.5B \
  --dtype auto \
  --dataset_name deepscaler_conversation.jsonl \
  --dataset_train_split train \
  --bf16 True \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --temperature 1 \
  --top_p 0.95 \
  --top_k 0 \
  --max_completion_length 600 \
  --max_length 3000 \
  --lmbda 1.0 \
  --logging_steps 1 \
  --report_to swanlab \
  --project from_r1_7b_to_1.5b \
  --seed 42 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type constant \
  --trust_remote_code True \
  --teacher_model_name_or_path /code/OnlineSFT/osft/outputs/OSFT-R1-Distill-Qwen-1.5B/OSFT/CE-dsr/1218_0155/global_step_340/actor/huggingface \
  --teacher_tokenizer_name_or_path /code/OnlineSFT/osft/outputs/OSFT-R1-Distill-Qwen-1.5B/OSFT/CE-dsr/1218_0155/global_step_340/actor/huggingface \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 30 \
  --output_dir trainer_output_cross_tokenizer_max_length_600 \
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
  --run_name "max_len_600" \
  # 需要考虑的term：output_dir， max_completion_length, run_name