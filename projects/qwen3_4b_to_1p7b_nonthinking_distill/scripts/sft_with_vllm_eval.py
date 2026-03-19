#!/usr/bin/env python
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Minimal SFT entrypoint with optional vLLM-based math eval callback.

from __future__ import annotations

import argparse
import gc
import inspect
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class VLLMEvalArguments:
    vllm_eval_enabled: bool = field(default=False, metadata={"help": "Enable vLLM eval callback."})
    vllm_eval_test_names: str = field(default="", metadata={"help": "Comma separated eval dataset names."})
    vllm_eval_test_paths: str = field(default="", metadata={"help": "Comma separated eval json/jsonl paths."})
    vllm_eval_test_samples: str = field(
        default="all", metadata={"help": "Comma separated sample limits per dataset; or 'all'."}
    )
    vllm_eval_temperature: float = field(default=0.0, metadata={"help": "vLLM eval temperature."})
    vllm_eval_n: int = field(default=1, metadata={"help": "Number of generations per prompt in eval."})
    vllm_eval_max_new_tokens: int = field(default=512, metadata={"help": "vLLM eval max new tokens."})
    vllm_eval_gpu: str = field(default="", metadata={"help": "CUDA_VISIBLE_DEVICES for vLLM eval engine."})
    vllm_eval_tensor_parallel_size: int = field(default=1, metadata={"help": "vLLM tensor parallel size."})
    vllm_eval_dtype: str = field(default="auto", metadata={"help": "vLLM dtype."})
    vllm_eval_max_model_len: int = field(default=8192, metadata={"help": "vLLM max model len."})
    vllm_eval_gpu_memory_utilization: float = field(default=0.9, metadata={"help": "vLLM GPU memory utilization."})


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


class SFTVLLMMathEvalCallback(TrainerCallback):
    def __init__(self, trainer: SFTTrainer, eval_args: VLLMEvalArguments):
        self.trainer = trainer
        self.eval_args = eval_args
        self._last_eval_step = -1

        try:
            from math_verify import parse, verify
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("vLLM eval callback requires `math-verify`.") from exc
        self._parse = parse
        self._verify = verify

        self._supports_enable_thinking = self._check_enable_thinking_support()
        self.all_test_data = self._load_all_eval_data()

    def _check_enable_thinking_support(self) -> bool:
        try:
            signature = inspect.signature(self.trainer.processing_class.apply_chat_template)
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            return "enable_thinking" in signature.parameters or has_var_kwargs
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [x.strip() for x in value.split(",") if x.strip()]

    @staticmethod
    def _extract_question(item: dict) -> str:
        for key in ("problem", "question", "prompt"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_gold(item: dict) -> str:
        for key in ("answer", "gold"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _load_all_eval_data(self) -> dict[str, list[dict]]:
        names = self._split_csv(self.eval_args.vllm_eval_test_names)
        paths = self._split_csv(self.eval_args.vllm_eval_test_paths)
        sample_limits = self._split_csv(self.eval_args.vllm_eval_test_samples)

        if not names or not paths:
            raise ValueError("vLLM eval enabled but eval names/paths are empty.")
        if len(names) != len(paths):
            raise ValueError("`vllm_eval_test_names` and `vllm_eval_test_paths` count mismatch.")
        if len(sample_limits) == 1:
            sample_limits = sample_limits * len(names)
        if len(sample_limits) != len(names):
            raise ValueError("`vllm_eval_test_samples` count mismatch.")

        all_data: dict[str, list[dict]] = {}
        rng = random.Random(42)
        for name, path, limit in zip(names, paths, sample_limits, strict=True):
            raw = load_dataset("json", data_files=path, split="train")
            rows = [row for row in raw]
            if limit.lower() != "all":
                max_n = min(len(rows), int(limit))
                if max_n < len(rows):
                    rows = rng.sample(rows, k=max_n)

            processed = []
            for item in rows:
                question = self._extract_question(item)
                gold = self._extract_gold(item)
                if not question or not gold:
                    continue
                processed.append({"question": question, "gold": gold})
            all_data[name] = processed
            logger.info(f"vLLM eval dataset '{name}' loaded: {len(processed)} samples")

        return all_data

    def _find_latest_checkpoint(self) -> str | None:
        out = Path(self.trainer.args.output_dir)
        if not out.exists():
            return None
        candidates = []
        for ckpt in out.glob("checkpoint-*"):
            if not ckpt.is_dir():
                continue
            try:
                step = int(ckpt.name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, ckpt))
        if not candidates:
            return None
        _, ckpt_path = max(candidates, key=lambda x: x[0])
        return str(ckpt_path)

    def _build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_enable_thinking:
            kwargs["enable_thinking"] = True
        return self.trainer.processing_class.apply_chat_template(messages, **kwargs)

    def _is_correct(self, gold: str, pred_text: str) -> bool:
        candidate = extract_last_boxed(pred_text) or pred_text
        try:
            gold_parsed = self._parse(gold)
            pred_parsed = self._parse(candidate)
            if len(gold_parsed) == 0 or len(pred_parsed) == 0:
                return False
            return bool(self._verify(gold_parsed, pred_parsed))
        except Exception:  # noqa: BLE001
            return False

    def _create_vllm_engine(self, model_path: str):
        from vllm import LLM

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.eval_args.vllm_eval_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.eval_args.vllm_eval_gpu

        try:
            engine = LLM(
                model=model_path,
                tensor_parallel_size=self.eval_args.vllm_eval_tensor_parallel_size,
                dtype=self.eval_args.vllm_eval_dtype,
                trust_remote_code=True,
                max_model_len=self.eval_args.vllm_eval_max_model_len,
                gpu_memory_utilization=self.eval_args.vllm_eval_gpu_memory_utilization,
            )
        finally:
            if original_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

        return engine

    def _run_eval_once(self, state_step: int) -> None:
        checkpoint_path = self._find_latest_checkpoint()
        if checkpoint_path is None:
            logger.warning("Skip vLLM eval: no checkpoint found under output_dir yet.")
            return

        from vllm import SamplingParams

        logger.info(f"Starting vLLM eval at step={state_step}, checkpoint={checkpoint_path}")
        eval_start = time.time()
        engine = self._create_vllm_engine(checkpoint_path)
        sampling_params = SamplingParams(
            n=self.eval_args.vllm_eval_n,
            temperature=self.eval_args.vllm_eval_temperature,
            max_tokens=self.eval_args.vllm_eval_max_new_tokens,
        )

        for name, data in self.all_test_data.items():
            if not data:
                continue
            prompts = [self._build_prompt(item["question"]) for item in data]
            outputs = engine.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

            correct = 0
            total_len = 0
            for item, output in zip(data, outputs, strict=True):
                any_correct = False
                for candidate in output.outputs:
                    total_len += len(candidate.token_ids)
                    if self._is_correct(item["gold"], candidate.text):
                        any_correct = True
                if any_correct:
                    correct += 1

            denom = max(len(data) * max(self.eval_args.vllm_eval_n, 1), 1)
            metrics = {
                f"eval/{name}_acc": correct / len(data),
                f"eval/{name}_avg_len": total_len / denom,
                f"eval/{name}_count": len(data),
                "step": state_step,
            }
            self.trainer.log(metrics)
            logger.info(
                f"vLLM eval '{name}': acc={metrics[f'eval/{name}_acc']:.4f}, "
                f"avg_len={metrics[f'eval/{name}_avg_len']:.1f}, n={len(data)}"
            )

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Finished vLLM eval at step={state_step} in {time.time() - eval_start:.1f}s")

    def on_save(self, args, state, control, **kwargs):
        if not self.trainer.accelerator.is_main_process:
            return control
        if state.global_step <= 0:
            return control
        if state.global_step == self._last_eval_step:
            return control

        self._run_eval_once(state.global_step)
        self._last_eval_step = state.global_step
        return control


def main(script_args, training_args, model_args, dataset_args, vllm_eval_args):
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if dataset_args.datasets and script_args.dataset_name:
        logger.warning("Both `datasets` and `dataset_name` are provided. Using `datasets` from config.")
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    if vllm_eval_args.vllm_eval_enabled:
        trainer.add_callback(SFTVLLMMathEvalCallback(trainer=trainer, eval_args=vllm_eval_args))
        logger.info("Enabled vLLM eval callback for SFT.")

    trainer.train()

    trainer.accelerator.print("Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig, VLLMEvalArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run SFT with optional vLLM eval callback", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, vllm_eval_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args, vllm_eval_args)
