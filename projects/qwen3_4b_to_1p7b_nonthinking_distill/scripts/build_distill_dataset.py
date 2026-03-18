#!/usr/bin/env python
"""
Build an offline distillation dataset by:
1) sampling prompts from a source JSONL dataset,
2) generating teacher responses with vLLM in non-thinking mode, and
3) keeping only math-verified correct responses.
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
    parser.add_argument("--solution_field", type=str, default="solution")
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
    solution_field: str,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0

    all_rows = _load_json_or_jsonl(input_jsonl)
    for idx, obj in enumerate(all_rows):
        question = obj.get(question_field)
        answer = obj.get(answer_field)
        solution = obj.get(solution_field)

        if not isinstance(question, str) or not isinstance(answer, str):
            skipped += 1
            continue

        rows.append(
            {
                "source_idx": idx,
                "question": question.strip(),
                "gold_answer": answer.strip(),
                "gold_solution": solution.strip() if isinstance(solution, str) else "",
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


def verify_by_answer_only(
    parse,
    verify,
    latex_cfg_cls,
    expr_cfg_cls,
    normalization_cfg_cls,
    gold_answer: str,
    pred_text: str,
) -> tuple[bool, bool, bool, str]:
    """
    Returns:
      (is_correct, gold_parse_failed, pred_parse_failed, error_message)
    """
    try:
        extraction_cfg = [
            latex_cfg_cls(
                normalization_config=normalization_cfg_cls(units=True),
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            ),
            expr_cfg_cls(),
        ]
        gold_parsed = parse(
            gold_answer,
            extraction_config=extraction_cfg,
            extraction_mode="first_match",
        )
        pred_parsed = parse(
            pred_text,
            extraction_config=extraction_cfg,
            extraction_mode="first_match",
        )
        gold_failed = len(gold_parsed) == 0
        pred_failed = len(pred_parsed) == 0
        if gold_failed or pred_failed:
            return False, gold_failed, pred_failed, ""
        return bool(verify(gold_parsed, pred_parsed)), False, False, ""
    except Exception as exc:  # noqa: BLE001
        return False, False, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

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
        solution_field=args.solution_field,
    )
    if not rows:
        raise ValueError(f"No valid rows found in {input_jsonl}")

    sample_size = min(args.sample_size, len(rows))
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, k=sample_size)

    print(f"[INFO] Loaded rows: {len(rows)} (skipped: {skipped})")
    print(f"[INFO] Sampled rows: {sample_size} (seed={args.seed})")
    print(f"[INFO] Teacher model: {args.teacher_model_path}")

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
        "input_schema": {
            "question_field": args.question_field,
            "answer_field": args.answer_field,
            "solution_field": args.solution_field,
        },
        "sampling_params": {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "generation": {
            "processed": 0,
            "correct": 0,
            "correct_rate": 0.0,
            "gold_parse_failed": 0,
            "pred_parse_failed": 0,
            "verification_errors": 0,
            "chat_template_enable_thinking_supported": None,
        },
        "outputs": {
            "all_generations_jsonl": str(all_path),
            "filtered_correct_jsonl": str(filtered_path),
            "stats_json": str(stats_path),
        },
    }

    with all_path.open("w", encoding="utf-8") as all_f, filtered_path.open("w", encoding="utf-8") as filtered_f:
        for start in range(0, sample_size, args.batch_size):
            batch = sampled[start : start + args.batch_size]
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
                is_correct, gold_parse_failed, pred_parse_failed, verifier_error = verify_by_answer_only(
                    parse=parse,
                    verify=verify,
                    latex_cfg_cls=LatexExtractionConfig,
                    expr_cfg_cls=ExprExtractionConfig,
                    normalization_cfg_cls=NormalizationConfig,
                    gold_answer=item["gold_answer"],
                    pred_text=selected_response,
                )

                stats["generation"]["processed"] += 1
                if gold_parse_failed:
                    stats["generation"]["gold_parse_failed"] += 1
                if pred_parse_failed:
                    stats["generation"]["pred_parse_failed"] += 1
                if verifier_error:
                    stats["generation"]["verification_errors"] += 1
                if is_correct:
                    stats["generation"]["correct"] += 1

                all_row = {
                    "source_idx": item["source_idx"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "gold_solution": item["gold_solution"],
                    "teacher_response": selected_response,
                    "is_correct": is_correct,
                    "gold_parse_failed": gold_parse_failed,
                    "pred_parse_failed": pred_parse_failed,
                    "verifier_error": verifier_error,
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

            processed = stats["generation"]["processed"]
            print(f"[INFO] Processed {processed}/{sample_size}")

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
