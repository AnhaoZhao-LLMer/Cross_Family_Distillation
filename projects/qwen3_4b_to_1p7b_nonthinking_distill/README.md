# Qwen3 4B -> 1.7B Base Offline Distillation

This project builds a filtered distillation dataset and then runs SFT.

## Paths (server)

- Teacher: `/code/models/Qwen3-4B/`
- Student: `/code/models/Qwen3-1.7B-Base/`
- Source dataset: `/code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/data/deepscaler/train.jsonl`
- Project root: `/code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill`

## Step 0: install dependencies

At minimum, ensure these packages are available in your runtime:

- `vllm`
- `transformers`
- `datasets`
- `accelerate`
- local `trl` source (this repo)

## Step 1: generate + filter distillation dataset

```bash
bash /code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill/scripts/run_generate_and_filter.sh
```

Outputs (under `data/`):

- `all_generations.jsonl`
- `filtered_correct_messages_10k.jsonl`
- `stats.json`

## Step 2: run SFT on Qwen3-1.7B-Base

```bash
bash /code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill/scripts/run_sft_qwen3_1p7b.sh
```

## Notes

- Prompt rendering uses `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` when supported.
- Verification uses `answer` field directly and compares to the extracted last `\boxed{...}` content.
- SFT dataset format is conversational `messages` JSONL.
