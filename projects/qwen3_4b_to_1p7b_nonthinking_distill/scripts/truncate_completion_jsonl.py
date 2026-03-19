#!/usr/bin/env python
"""
Truncate assistant completion tokens in conversational JSONL.

Expected row format:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truncate assistant completion tokens in JSONL.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_completion_tokens", type=int, required=True)
    parser.add_argument("--assistant_index", type=int, default=1, help="Index of assistant message in messages list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    total = 0
    kept = 0
    truncated = 0
    skipped = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) <= args.assistant_index:
                skipped += 1
                continue

            assistant = messages[args.assistant_index]
            if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
                skipped += 1
                continue

            content = assistant.get("content")
            if not isinstance(content, str):
                skipped += 1
                continue

            ids = tokenizer(content, add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_completion_tokens:
                ids = ids[: args.max_completion_tokens]
                truncated += 1
            else:
                kept += 1

            assistant["content"] = tokenizer.decode(ids, skip_special_tokens=False)
            obj["messages"] = messages
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] total={total} kept={kept} truncated={truncated} skipped={skipped}")
    print(f"[INFO] output={args.output_jsonl}")


if __name__ == "__main__":
    main()
