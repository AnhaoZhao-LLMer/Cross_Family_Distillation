#!/usr/bin/env python
"""
Build an offline distillation dataset by:
1) sampling prompts from a source JSONL dataset,
2) generating teacher responses with vLLM in non-thinking mode, and
3) keeping only responses whose last boxed answer matches the gold answer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and filter distillation data for Qwen3 offline KD.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/data/deepscaler/train.jsonl",
        help="Input dataset path. Supports JSONL (one object per line) or JSON array file.",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="/code/models/Qwen3-4B/",
        help="Teacher model path (vLLM/Transformers-compatible).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/code/on_policy_distillation/trl/projects/qwen3_4b_to_1p7b_nonthinking_distill/data",
        help="Output directory for all_generations.jsonl, filtered_correct_messages_10k.jsonl, stats.json",
    )
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question_field", type=str, default="problem")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument(
        "--boxed_compare_mode",
        type=str,
        choices=["string", "math"],
        default="math",
        help="How to compare boxed answer with gold answer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1, help="Number of generations per prompt.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--all_generations_filename", type=str, default="all_generations.jsonl")
    parser.add_argument("--filtered_filename", type=str, default="filtered_correct_messages_10k.jsonl")
    parser.add_argument("--stats_filename", type=str, default="stats.json")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing jsonl files if they exist.",
    )
    return parser.parse_args()


def _load_json_or_jsonl(input_path: Path) -> list[dict]:
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []

    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_source_rows(
    input_jsonl: Path,
    question_field: str,
    answer_field: str,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0

    all_rows = _load_json_or_jsonl(input_jsonl)
    for idx, obj in enumerate(all_rows):
        question = obj.get(question_field)
        answer = obj.get(answer_field)

        if not isinstance(question, str) or not isinstance(answer, str):
            skipped += 1
            continue

        rows.append(
            {
                "source_idx": idx,
                "question": question.strip(),
                "gold_answer": answer.strip(),
            }
        )

    return rows, skipped


def build_prompt(tokenizer, question: str) -> tuple[str, bool]:
    messages = [{"role": "user", "content": question}]
    apply_fn = tokenizer.apply_chat_template
    try:
        signature = inspect.signature(apply_fn)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        supports_enable_thinking = ("enable_thinking" in signature.parameters) or has_var_kwargs
    except (TypeError, ValueError):
        supports_enable_thinking = False
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if supports_enable_thinking:
        kwargs["enable_thinking"] = False
    prompt = apply_fn(messages, **kwargs)
    return prompt, supports_enable_thinking


def extract_last_boxed(text: str) -> str:
    if not text:
        return ""
    pos = 0
    last = ""
    while True:
        idx = text.find("\\boxed", pos)
        if idx < 0:
            break
        brace_start = text.find("{", idx)
        if brace_start < 0:
            pos = idx + 6
            continue

        depth = 0
        for i in range(brace_start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace_start + 1 : i].strip()
                    break
        pos = idx + 6
    return last


def normalize_answer(text: str) -> str:
    s = text.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\u2212", "-")
    s = "".join(s.split())
    return s


def load_existing_progress(all_path: Path, filtered_path: Path) -> tuple[set[int], set[int]]:
    processed_source_idx: set[int] = set()
    correct_source_idx: set[int] = set()

    if all_path.exists():
        with all_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_idx = obj.get("source_idx")
                if isinstance(source_idx, int):
                    processed_source_idx.add(source_idx)
                    if bool(obj.get("is_correct")):
                        correct_source_idx.add(source_idx)

    # If filtered exists but all doesn't, we still rely on all file for source_idx dedup.
    # Keeping this check for visibility only.
    if filtered_path.exists() and not all_path.exists():
        print(f"[WARN] {filtered_path} exists but {all_path} does not. Resume dedup may be incomplete.")

    return processed_source_idx, correct_source_idx


def compare_boxed_answer(
    gold_answer: str,
    pred_boxed: str,
    compare_mode: str,
    parse=None,
    verify=None,
) -> bool:
    if not pred_boxed:
        return False

    if compare_mode == "string":
        return normalize_answer(pred_boxed) == normalize_answer(gold_answer)

    # compare_mode == "math"
    if parse is None or verify is None:
        raise ValueError("math compare mode requires math_verify parse/verify functions.")
    try:
        gold_parsed = parse(gold_answer)
        pred_parsed = parse(pred_boxed)
        if len(gold_parsed) == 0 or len(pred_parsed) == 0:
            return False
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    args = parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parse = None
    verify = None
    if args.boxed_compare_mode == "math":
        try:
            from math_verify import parse as mv_parse, verify as mv_verify
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "boxed_compare_mode=math requires math_verify. Install it with `pip install math-verify`."
            ) from exc
        parse = mv_parse
        verify = mv_verify

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / args.all_generations_filename
    filtered_path = output_dir / args.filtered_filename
    stats_path = output_dir / args.stats_filename

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")

    rows, skipped = load_source_rows(
        input_jsonl=input_jsonl,
        question_field=args.question_field,
        answer_field=args.answer_field,
    )
    if not rows:
        raise ValueError(f"No valid rows found in {input_jsonl}")

    sample_size = min(args.sample_size, len(rows))
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, k=sample_size)
    sampled_source_idx = {item["source_idx"] for item in sampled}

    existing_processed_idx: set[int] = set()
    existing_correct_idx: set[int] = set()
    if args.resume:
        existing_processed_idx, existing_correct_idx = load_existing_progress(all_path, filtered_path)
        # Keep only idx that belong to this sampled subset.
        existing_processed_idx = existing_processed_idx.intersection(sampled_source_idx)
        existing_correct_idx = existing_correct_idx.intersection(sampled_source_idx)
    to_process = [item for item in sampled if item["source_idx"] not in existing_processed_idx]

    print(f"[INFO] Loaded rows: {len(rows)} (skipped: {skipped})")
    print(f"[INFO] Sampled rows: {sample_size} (seed={args.seed})")
    print(f"[INFO] Teacher model: {args.teacher_model_path}")
    if args.resume:
        print(f"[INFO] Resume enabled: already_done_in_sample={len(existing_processed_idx)}")
    print(f"[INFO] Remaining to process: {len(to_process)}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.teacher_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
    )

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_jsonl),
        "teacher_model_path": args.teacher_model_path,
        "total_valid_source_rows": len(rows),
        "total_skipped_source_rows": skipped,
        "sample_size_requested": args.sample_size,
        "sample_size_actual": sample_size,
        "seed": args.seed,
        "resume": args.resume,
        "existing_processed_in_sample": len(existing_processed_idx),
        "input_schema": {
            "question_field": args.question_field,
            "answer_field": args.answer_field,
        },
        "boxed_compare_mode": args.boxed_compare_mode,
        "sampling_params": {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "generation": {
            "processed": len(existing_processed_idx),
            "correct": len(existing_correct_idx),
            "correct_rate": 0.0,
            "boxed_not_found": 0,
            "chat_template_enable_thinking_supported": None,
        },
        "outputs": {
            "all_generations_jsonl": str(all_path),
            "filtered_correct_jsonl": str(filtered_path),
            "stats_json": str(stats_path),
        },
    }

    all_mode = "a" if args.resume and all_path.exists() else "w"
    filtered_mode = "a" if args.resume and filtered_path.exists() else "w"

    try:
        with all_path.open(all_mode, encoding="utf-8") as all_f, filtered_path.open(
            filtered_mode, encoding="utf-8"
        ) as filtered_f:
            for start in range(0, len(to_process), args.batch_size):
                batch = to_process[start : start + args.batch_size]
                prompts = []
                support_flags = []

                for item in batch:
                    prompt, supports_enable_thinking = build_prompt(tokenizer, item["question"])
                    prompts.append(prompt)
                    support_flags.append(supports_enable_thinking)

                if stats["generation"]["chat_template_enable_thinking_supported"] is None and support_flags:
                    stats["generation"]["chat_template_enable_thinking_supported"] = support_flags[0]

                outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

                for item, output in zip(batch, outputs, strict=True):
                    completions = [candidate.text for candidate in output.outputs]
                    if not completions:
                        completions = [""]

                    # Keep logic simple: only evaluate the first completion.
                    selected_response = completions[0]
                    pred_boxed = extract_last_boxed(selected_response)
                    if not pred_boxed:
                        stats["generation"]["boxed_not_found"] += 1
                    is_correct = compare_boxed_answer(
                        gold_answer=item["gold_answer"],
                        pred_boxed=pred_boxed,
                        compare_mode=args.boxed_compare_mode,
                        parse=parse,
                        verify=verify,
                    )

                    stats["generation"]["processed"] += 1
                    if is_correct:
                        stats["generation"]["correct"] += 1

                    all_row = {
                        "source_idx": item["source_idx"],
                        "question": item["question"],
                        "gold_answer": item["gold_answer"],
                        "teacher_response": selected_response,
                        "teacher_boxed_answer": pred_boxed,
                        "is_correct": is_correct,
                    }
                    all_f.write(json.dumps(all_row, ensure_ascii=False) + "\n")

                    if is_correct:
                        filtered_row = {
                            "messages": [
                                {"role": "user", "content": item["question"]},
                                {"role": "assistant", "content": selected_response},
                            ],
                            "meta": {
                                "source_idx": item["source_idx"],
                                "gold_answer": item["gold_answer"],
                            },
                        }
                        filtered_f.write(json.dumps(filtered_row, ensure_ascii=False) + "\n")

                # Periodic durability for Ctrl+C and crashes.
                all_f.flush()
                filtered_f.flush()
                processed = stats["generation"]["processed"]
                print(f"[INFO] Processed {processed}/{sample_size}")
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user (Ctrl+C). Saving partial progress and stats...")

    if stats["generation"]["processed"] > 0:
        stats["generation"]["correct_rate"] = (
            stats["generation"]["correct"] / stats["generation"]["processed"]
        )

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[INFO] Finished dataset generation and filtering.")
    print(f"[INFO] all_generations: {all_path}")
    print(f"[INFO] filtered_correct: {filtered_path}")
    print(
        "[INFO] summary: "
        f"processed={stats['generation']['processed']} "
        f"correct={stats['generation']['correct']} "
        f"correct_rate={stats['generation']['correct_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
