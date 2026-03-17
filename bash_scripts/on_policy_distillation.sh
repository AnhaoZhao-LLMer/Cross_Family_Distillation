# accelerate launch \
#   --config_file examples/accelerate_configs/multi_gpu.yaml trl/experimental/gold/gold.py \
CUDA_VISIBLE_DEVICES=0 python trl/experimental/gold/gold.py \
  --model_name_or_path /code/models/Qwen3-1.7B-Base \
  --dtype auto \
  --dataset_name deepscaler_conversation.jsonl \
  --dataset_train_split train \
  --bf16 True \
  --learning_rate 5e-7 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 0 \
  --max_completion_length 2000 \
  --max_length 3000 \
  --lmbda 1 \
  --beta 0.0 \
  --logging_steps 1 \
  --report_to swanlab \
  --project from_r1_7b_to_1.5b \
  --seed 42 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type constant \
  --trust_remote_code True \
  --teacher_model_name_or_path /code/models/Qwen3-4B \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 30 \
  --output_dir trainer_output \
  --use_vllm True \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.2 \
  --vllm_tensor_parallel_size 1 \
  # --vllm_enable_sleep_mode True \
  # --vllm_server_port 8000

  # --eval_strategy steps \
  # --eval_steps 20 \
  # --attn_implementation flash_attention_2 \
  # --gradient_checkpointing \
  # --use_uld_loss \
  # --use_extended_uld \
  # --uld_use_hybrid_loss \
  # --uld_crossentropy_weight 0.0 \
  # --uld_distillation_weight 1.0 \
  # --uld_student_temperature 1.0 \
  # --uld_teacher_temperature 1.0 \
  # --uld_hybrid_unmatched_weight 1.0 \
  # --uld_hybrid_matched_weight 1.0 \
  # --push_to_hub \
  # --hub_model_id <your-username>/Qwen3-4B-GKD-Tulu \