#!/bin/bash
set -euo pipefail

REPO_ROOT="/code/on_policy_distillation/trl"
PROJECT_ROOT="/code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill"

INPUT_JSONL="${PROJECT_ROOT}/data/all_generations.jsonl"
OUTPUT_JSONL="${PROJECT_ROOT}/data/all_generations_trunc_c1024_messages.jsonl"
TOKENIZER_PATH="/code/models/Qwen3-1.7B-Base/"
MAX_COMPLETION_TOKENS=1024

cd "${REPO_ROOT}"

python "${PROJECT_ROOT}/scripts/truncate_completion_jsonl.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --max_completion_tokens "${MAX_COMPLETION_TOKENS}" \
  --assistant_index 1 \
  --no-drop_incorrect
