# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import inspect
import json
import time
import warnings
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback


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


def extract_after_hashes(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    return ""


class GOLDVLLMMathEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.args = trainer.args
        self.eval_steps = int(getattr(self.args, "eval_steps", 0) or 0)
        self._last_eval_step = -1
        self._initial_eval_done = False

        names_str = getattr(self.args, "eval_test_names", "")
        paths_str = getattr(self.args, "eval_test_paths", "")
        samples_str = getattr(self.args, "eval_test_samples", "all")

        if not names_str or not paths_str:
            self.all_test_data: dict[str, list[dict]] = {}
            return

        try:
            from math_verify import parse, verify
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("GOLD vLLM eval callback requires `math-verify`.") from exc
        self._parse = parse
        self._verify = verify

        self.names = [n.strip() for n in names_str.split(",") if n.strip()]
        self.paths = [p.strip() for p in paths_str.split(",") if p.strip()]
        sample_list = [s.strip() for s in samples_str.split(",") if s.strip()]
        if len(sample_list) == 1 and self.names:
            sample_list = sample_list * len(self.names)
        self.sample_limits = sample_list

        if len(self.names) != len(self.paths) or len(self.names) != len(self.sample_limits):
            raise ValueError("eval_test_names, eval_test_paths, eval_test_samples counts must match.")

        self._supports_enable_thinking = self._check_enable_thinking_support()
        self._enable_thinking = getattr(self.args, "enable_thinking", None)

        self.eval_output_dir = Path(self.args.output_dir) / "eval_outputs"
        if self.trainer.accelerator.is_main_process:
            self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Load on all ranks, so colocate + TP>1 can safely run generation together.
        self.all_test_data = {}
        for name, path, limit in zip(self.names, self.paths, self.sample_limits, strict=True):
            self.all_test_data[name] = self._prepare_dataset(name, path, limit)

    def _check_enable_thinking_support(self) -> bool:
        try:
            signature = inspect.signature(self.trainer.processing_class.apply_chat_template)
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            return "enable_thinking" in signature.parameters or has_var_kwargs
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _extract_question(name: str, item: dict) -> str:
        if name.lower() == "gsm8k":
            question = item.get("question", "")
            return question.strip() if isinstance(question, str) else ""

        for key in ("problem", "question", "prompt"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_gold(name: str, item: dict) -> str:
        if name.lower() == "gsm8k":
            answer = item.get("answer", "")
            if isinstance(answer, str):
                answer = answer.strip()
                return extract_after_hashes(answer) or answer
            return ""

        for key in ("answer", "gold", "solution"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                return extract_after_hashes(value) or value
        return ""

    def _prepare_dataset(self, name: str, path: str, limit: str) -> list[dict]:
        raw_dataset = load_dataset("json", data_files=path, split="train")
        if limit.lower() != "all":
            try:
                max_n = min(len(raw_dataset), int(limit))
                raw_dataset = raw_dataset.shuffle(seed=42).select(range(max_n))
            except ValueError:
                pass

        processed = []
        for item in raw_dataset:
            question = self._extract_question(name, item)
            gold = self._extract_gold(name, item)
            if question and gold:
                processed.append({"question": question, "gold": gold})
        return processed

    def _build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_enable_thinking and self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking
        prompt = self.trainer.processing_class.apply_chat_template(messages, **kwargs)

        bos_token = getattr(self.trainer.processing_class, "bos_token", None)
        if isinstance(bos_token, str) and bos_token and prompt.startswith(bos_token):
            prompt = prompt[len(bos_token) :]
        return prompt

    def _is_correct(self, gold: str, pred_text: str) -> bool:
        candidate = extract_last_boxed(pred_text) or extract_after_hashes(pred_text) or pred_text
        try:
            gold_parsed = self._parse(gold)
            pred_parsed = self._parse(candidate)
            if len(gold_parsed) == 0 or len(pred_parsed) == 0:
                return False
            return bool(self._verify(gold_parsed, pred_parsed))
        except Exception:  # noqa: BLE001
            return False

    def _needs_all_ranks_for_eval(self) -> bool:
        return bool(
            getattr(self.trainer, "use_vllm", False)
            and getattr(self.trainer, "vllm_mode", "") == "colocate"
            and int(getattr(self.trainer, "vllm_tensor_parallel_size", 1)) > 1
        )

    def _generate_grouped_candidates(self, prompts: list[str], eval_n: int, eval_sampling_params):
        is_main = self.trainer.accelerator.is_main_process
        if self.trainer.vllm_mode == "server":
            if not is_main:
                return []
            top_p = getattr(self.args, "eval_top_p", getattr(self.args, "top_p", 1.0))
            top_k = getattr(self.args, "eval_top_k", getattr(self.args, "top_k", -1))
            min_p = getattr(self.args, "eval_min_p", getattr(self.args, "min_p", 0.0))
            repetition_penalty = getattr(
                self.args, "eval_repetition_penalty", getattr(self.args, "repetition_penalty", 1.0)
            )
            response = self.trainer.vllm_client.generate(
                prompts=prompts,
                n=eval_n,
                repetition_penalty=repetition_penalty,
                temperature=self.args.eval_temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=eval_sampling_params.max_tokens,
                structured_outputs_regex=getattr(self.trainer, "vllm_structured_outputs_regex", None),
            )
            completion_ids = response["completion_ids"]
            expected = len(prompts) * eval_n
            if len(completion_ids) != expected:
                warnings.warn(
                    f"vLLM server returned {len(completion_ids)} completions, expected {expected}.",
                    stacklevel=2,
                )

            grouped = []
            idx = 0
            for _ in prompts:
                candidates = []
                for _ in range(eval_n):
                    if idx < len(completion_ids):
                        token_ids = completion_ids[idx]
                        text = self.trainer.processing_class.decode(token_ids, skip_special_tokens=False)
                        candidates.append((text, len(token_ids)))
                    idx += 1
                grouped.append(candidates)
            return grouped

        all_outputs = self.trainer.vllm_engine.generate(prompts, sampling_params=eval_sampling_params, use_tqdm=is_main)
        if not is_main:
            return []

        grouped = []
        for output in all_outputs:
            candidates = [(candidate.text, len(candidate.token_ids)) for candidate in output.outputs]
            grouped.append(candidates)
        return grouped

    def _do_eval(self, state) -> None:
        if not getattr(self, "all_test_data", None):
            return
        if not getattr(self.trainer, "use_vllm", False):
            return

        is_main = self.trainer.accelerator.is_main_process
        needs_all_ranks = self._needs_all_ranks_for_eval()
        if not is_main and not needs_all_ranks:
            return

        from vllm import SamplingParams

        eval_temp = getattr(self.args, "eval_temperature", 0.0)
        eval_n = max(1, int(getattr(self.args, "eval_n", 1)))
        eval_max_tokens = getattr(self.args, "eval_max_new_tokens", None)
        if eval_max_tokens is None:
            eval_max_tokens = getattr(self.args, "max_completion_length", 512)
        eval_top_p = getattr(self.args, "eval_top_p", getattr(self.args, "top_p", 1.0))
        eval_top_k = getattr(self.args, "eval_top_k", getattr(self.args, "top_k", -1))
        eval_sampling_params = SamplingParams(
            n=eval_n,
            temperature=eval_temp,
            top_p=eval_top_p,
            top_k=eval_top_k,
            max_tokens=eval_max_tokens,
        )

        is_colocate = self.trainer.vllm_mode == "colocate"
        if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
            self.trainer.vllm_engine.wake_up()

        details_rows = []
        try:
            for name, data in self.all_test_data.items():
                if not data:
                    continue

                prompts = [self._build_prompt(item["question"]) for item in data]
                grouped_candidates = self._generate_grouped_candidates(prompts, eval_n, eval_sampling_params)

                if not is_main:
                    continue

                start_time = time.time()
                correct_count = 0
                total_len = 0
                for sample_idx, (item, candidates) in enumerate(zip(data, grouped_candidates, strict=True)):
                    selected_text = ""
                    selected_len = 0
                    any_correct = False

                    if candidates:
                        selected_text, selected_len = candidates[0]

                    for pred_text, pred_len in candidates:
                        total_len += pred_len
                        try:
                            is_correct = self._is_correct(item["gold"], pred_text)
                        except Exception:  # noqa: BLE001
                            is_correct = False
                        if is_correct and not any_correct:
                            any_correct = True
                            selected_text = pred_text
                            selected_len = pred_len

                    if any_correct:
                        correct_count += 1

                    details_rows.append(
                        {
                            "step": int(state.global_step),
                            "dataset_name": name,
                            "sample_idx": int(sample_idx),
                            "question": item["question"],
                            "gold_answer": item["gold"],
                            "prediction_text": selected_text,
                            "is_correct": bool(any_correct),
                            "prediction_token_len": int(selected_len),
                            "model_path": str(getattr(self.trainer, "model_name_or_path", "")),
                        }
                    )

                if len(data) > 0:
                    acc = correct_count / len(data)
                    avg_len = total_len / max(1, len(data) * eval_n)
                    self.trainer.log(
                        {
                            f"eval/{name}_acc": acc,
                            f"eval/{name}_avg_len": avg_len,
                            f"eval/{name}_count": len(data),
                            "step": state.global_step,
                        }
                    )
                    print(
                        f"[GOLD Eval] {name} step={state.global_step} "
                        f"acc={acc:.4f} avg_len={avg_len:.1f} n={len(data)} "
                        f"time={time.time() - start_time:.1f}s"
                    )

            if is_main:
                details_path = self.eval_output_dir / f"eval_step_{int(state.global_step)}.jsonl"
                try:
                    with details_path.open("w", encoding="utf-8") as f:
                        for row in details_rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Failed to write eval detail file {details_path}: {exc}", stacklevel=2)
        finally:
            if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
                self.trainer.vllm_engine.sleep(level=2)

    def on_step_begin(self, args, state, control, **kwargs):
        if not getattr(self, "all_test_data", None):
            return control
        if self._initial_eval_done:
            return control

        self._do_eval(state)
        self._initial_eval_done = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not getattr(self, "all_test_data", None):
            return control
        if self.eval_steps <= 0:
            return control
        if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
            return control
        if state.global_step == self._last_eval_step:
            return control

        self._do_eval(state)
        self._last_eval_step = state.global_step
        return control
