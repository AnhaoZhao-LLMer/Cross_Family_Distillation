#!/usr/bin/env python
"""
Truncate assistant completion tokens in conversational JSONL.

Supported input row formats:
1) Conversational:
   {
     "messages": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
2) All-generations style:
   {
     "source_idx": ...,
     "question": "...",
     "teacher_response": "...",
     "gold_answer": "...",      # optional
     "is_correct": true/false   # optional
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
    parser.add_argument(
        "--drop_incorrect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When input is all-generations style and has `is_correct`, drop rows with is_correct=false.",
    )
    return parser.parse_args()


def to_messages_obj(obj: dict, assistant_index: int) -> dict | None:
    """Convert supported input row into a messages-format object."""
    messages = obj.get("messages")
    if isinstance(messages, list) and len(messages) > assistant_index:
        assistant = messages[assistant_index]
        if isinstance(assistant, dict) and assistant.get("role") == "assistant":
            return obj
        return None

    # all_generations style -> convert
    question = obj.get("question")
    teacher_response = obj.get("teacher_response")
    if isinstance(question, str) and isinstance(teacher_response, str):
        out = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": teacher_response},
            ],
            "meta": {
                "source_idx": obj.get("source_idx"),
                "gold_answer": obj.get("gold_answer"),
            },
        }
        return out

    return None


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

            if args.drop_incorrect and "is_correct" in obj and not bool(obj.get("is_correct")):
                skipped += 1
                continue

            converted = to_messages_obj(obj, args.assistant_index)
            if converted is None:
                skipped += 1
                continue
            obj = converted

            messages = obj["messages"]
            assistant = messages[args.assistant_index]

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
