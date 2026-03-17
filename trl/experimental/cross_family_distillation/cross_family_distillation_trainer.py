import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate.utils import broadcast_object_list, gather_object

# 请根据你的实际目录结构调整以下导入
from ..gold import GOLDTrainer 
from ...trainer.utils import empty_cache
from ...extras.profiling import profiling_decorator
from ...import_utils import is_vllm_available
from trl.experimental.cross_family_distillation.eval_callbacks import VLLMMathEvalCallback

if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

import warnings

from collections.abc import Callable

from accelerate import PartialState

from datasets import Dataset, IterableDataset

from transformers.feature_extraction_utils import FeatureExtractionMixin

from transformers.image_processing_utils import BaseImageProcessor

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase



from ...data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset





if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

class CrossFamilyPolicyDistillTrainer(GOLDTrainer):
    """
    Cross-tokenizer / cross-family on-policy distillation trainer.
    
    Supports two modes via `args.distill_method`:
    1. 'seq_level_kd': Sequence-level Reverse KL Distillation.
    2. 'rl_based': GRPO-style Reinforcement Learning with group-relative advantage.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        
        # 加载 Teacher Tokenizer
        if not hasattr(args, "teacher_tokenizer_name_or_path") or args.teacher_tokenizer_name_or_path is None:
            raise ValueError("args.teacher_tokenizer_name_or_path must be set for cross-family policy distillation.")
            
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer_name_or_path)
        if not hasattr(self.teacher_tokenizer, "pad_token") or self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        # 冻结 Teacher 模型
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        
        # 🌟 依赖注入：直接把 self 扔进去！
        if getattr(args, "eval_test_names", None) and getattr(args, "eval_test_paths", None):
            self.add_callback(VLLMMathEvalCallback(self))

    @torch.no_grad()
    def _teacher_logp_of_completion(self, prompt_texts, completion_texts):
        device = self.accelerator.device
        
        # 1️⃣ tokenize prompt
        prompt_tok = self.teacher_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_tok.input_ids.to(device)
        prompt_attn = prompt_tok.attention_mask.to(device)

        # 2️⃣ tokenize completion
        completion_tok = self.teacher_tokenizer(
            completion_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False,
        )
        completion_ids = completion_tok.input_ids.to(device)
        completion_attn = completion_tok.attention_mask.to(device)

        # 3️⃣ 拼接
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_attn, completion_attn], dim=1)

        # 4️⃣ teacher forward
        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T, V)
        logprobs = F.log_softmax(logits, dim=-1)

        # next-token logprob
        target = input_ids[:, 1:]
        token_logp = logprobs[:, :-1, :].gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)

        # 5️⃣ completion mask（根据 prompt 固定长度）
        B, Tm1 = token_logp.shape
        idx = torch.arange(Tm1, device=device).unsqueeze(0)
        start = prompt_ids.shape[1] - 1  # completion logprob start index
        
        completion_mask = (idx >= start).to(token_logp.dtype)
        # 去掉 padding
        completion_mask = completion_mask * attention_mask[:, 1:].to(token_logp.dtype)

        # 6️⃣ sequence logprob
        teacher_logp = (token_logp * completion_mask).sum(dim=1)

        return teacher_logp

    def _student_logp_of_completion(self, student_logits, labels):
        """
        Compute log πθ(y|q) on the *completion* tokens only.
        labels: (B, T) with -100 for non-loss positions.
        """
        logprobs = F.log_softmax(student_logits, dim=-1)
        target = labels[:, 1:].contiguous()  # (B, T-1), -100 for ignore
        lp = logprobs[:, :-1, :]  # (B, T-1, V)

        mask = (target != -100)
        target_clamped = target.clone()
        target_clamped[~mask] = 0
        tok_lp = lp.gather(-1, target_clamped.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp * mask.to(tok_lp.dtype)
        seq_lp = tok_lp.sum(dim=1)  # (B,)
        
        return seq_lp, tok_lp, mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1) student forward (with grad)
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits

        # 2) prompt/completion texts (for teacher scoring)
        if "original_prompt_text" not in inputs or "original_completion_text" not in inputs:
            raise ValueError("Need original_prompt_text/original_completion_text for cross-family teacher scoring.")
        
        prompt_texts = inputs["original_prompt_text"]
        completion_texts = inputs["original_completion_text"]

        pad = self.processing_class.pad_token
        prompt_texts = [p.replace(pad, "") for p in prompt_texts]
        completion_texts = [c.replace(pad, "") for c in completion_texts]
        
        # 3) teacher score (no grad) - Sequence Logprob
        teacher_logp = self._teacher_logp_of_completion(prompt_texts, completion_texts)  # (B*G,)

        # 4) student logp on completion (current policy)
        labels = inputs["labels"]
        student_logp, tok_lp, mask = self._student_logp_of_completion(logits, labels)

        # 5) old policy logp (snapshot)
        with torch.no_grad():
            old_student_logp = student_logp.detach()

        # 6) Advantage Computation (分流逻辑)
        with torch.no_grad():
            comp_len = mask.sum(dim=1).clamp_min(1)  # avoid div-by-zero
            method = getattr(self.args, "distill_method", "rl_based")

            if method == "seq_level_kd":
                # 模式 1：Seq-level Reverse KL Distillation
                adv = 100 * (teacher_logp - old_student_logp) / comp_len
                
            elif method == "rl_based":
                # 模式 2：RL-based GRPO
                raw_reward = teacher_logp / comp_len
                
                G = getattr(self.args, "num_generations", 4)
                B_times_G = raw_reward.shape[0]
                
                if B_times_G % G != 0:
                    raise ValueError(f"Batch size {B_times_G} 必须能被 num_generations={G} 整除")
                
                B = B_times_G // G
                
                # 折叠计算组内统计量
                reward_grouped = raw_reward.view(B, G)
                mean_reward = reward_grouped.mean(dim=1, keepdim=True)
                
                adv_grouped = reward_grouped - mean_reward
                
                # 结合 Config 中的 normalize_advantage 参数
                if getattr(self.args, "normalize_advantage", True):
                    std_reward = reward_grouped.std(dim=1, keepdim=True)
                    adv_grouped = adv_grouped / (std_reward + 1e-4)

                adv = adv_grouped.view(-1)
            else:
                raise ValueError(f"Unknown distill_method: {method}. Use 'seq_level_kd' or 'rl_based'.")
            
        # 7) Token-level Policy Gradient Loss
        adv_expanded = adv.unsqueeze(1)  # (B*G, 1)
        pg_loss = -(adv_expanded * tok_lp).sum() / (mask.sum() + 1e-6)

        empty_cache()
        return (pg_loss, outputs) if return_outputs else pg_loss

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        device = self.accelerator.device

        # 🌟 1. 根据 method 动态获取生成数量 G
        method = getattr(self.args, "distill_method", "rl_based")
        if method == "seq_level_kd":
            G = 1
        elif method == "rl_based":
            G = getattr(self.args, "num_generations", 4)
        else:
            raise ValueError(f"Unknown distill_method: {method}.")
        
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm]

        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=False,
        )

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        top_p = getattr(self.args, "top_p", 1.0)
        repetition_penalty = getattr(self.args, "repetition_penalty", 1.0)
        min_p = getattr(self.args, "min_p", 0.0)

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=G,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    structured_outputs_regex=getattr(self, "vllm_structured_outputs_regex", None),
                )["completion_ids"]
            else:
                completion_ids = [None] * (len(all_prompts_text) * G)
                
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm) * G,
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm) * G,
            )
            completion_ids = completion_ids[process_slice]
            
        elif self.vllm_mode == "colocate":
            
            vllm_regex = getattr(self, "vllm_structured_outputs_regex", None)
            structured_outputs = StructuredOutputsParams(backend="outlines", regex=vllm_regex) if vllm_regex else None
                
            sampling_params = SamplingParams(
                n=G,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                structured_outputs=structured_outputs,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size * G, (local_rank_in_group + 1) * orig_size * G)
                completion_ids = completion_ids[tp_slice]

            if getattr(self, "vllm_enable_sleep_mode", False):
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # 🌟 3. 将 Prompt 复制 G 份对齐
        repeated_prompts_for_vllm = [p for p in prompts_text_for_vllm for _ in range(G)]
        repeated_prompts_with_special = [p for p in prompts_text_with_special for _ in range(G)]
        
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        prompt_tokenized = self.processing_class(
            repeated_prompts_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=True,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        # Pad completions
        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat([
                    completion_tensor,
                    torch.full((padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype),
                ])
                padded_completion_ids_list.append(padded_tensor)
            else:
                padded_completion_ids_list.append(completion_tensor)

        padded_completion_ids = torch.stack(padded_completion_ids_list)

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)
        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        prompt_lengths = prompt_ids.shape[1]
        new_labels[:, :prompt_lengths] = -100

        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, repeated_prompts_with_special, completion_texts
    
    def _prepare_dataset_with_original_text(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """
        Prepare dataset while preserving original text for cross-tokenizer distillation.
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            # Convert the dataset to ChatML if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            column_names = next(iter(dataset)).keys()
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns="conversations" if "conversations" in column_names else None,
                **map_kwargs,
            )

            # Apply the chat template if needed and preserve original text
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                        example["text"] = example["text"] + eos_token
                    elif "completion" in example and not example["completion"].endswith(eos_token):
                        example["completion"] = example["completion"] + eos_token
                    return example

                dataset = dataset.map(
                    add_eos,
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                    **map_kwargs,
                )

            # Tokenize the dataset while preserving original text
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset (preserving original text)"

            def tokenize_with_original_text(example, processing_class, dataset_text_field, assistant_only_loss):
                """Modified tokenization function that preserves original text."""
                result = {}
                if "prompt" in example:  # prompt-completion case
                    # Store original text
                    result["original_prompt_text"] = example["prompt"]
                    result["original_completion_text"] = example["completion"]

                    if is_conversational(example):
                        prompt_ids = processing_class.apply_chat_template(
                            example["prompt"], return_dict=False, **example.get("chat_template_kwargs", {})
                        )
                        prompt_completion_ids = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"],
                            return_dict=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                    else:
                        prompt_ids = processing_class(text=example["prompt"]).input_ids
                        prompt_completion_ids = processing_class(
                            text=example["prompt"] + example["completion"]
                        ).input_ids

                    # Check if the tokenized prompt starts with the tokenized prompt+completion
                    if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                        warnings.warn(
                            "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                            "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                            "token handling. Verify that the tokenizer is processing text consistently.",
                            stacklevel=2,
                        )

                    # Create a completion mask
                    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                    result.update(
                        {
                            "input_ids": prompt_completion_ids,
                            "completion_mask": completion_mask,
                            "attention_mask": [1] * len(prompt_completion_ids),  # Add attention mask
                        }
                    )

                else:  # language modeling or conversational case
                    if is_conversational(example):
                        # For conversational data (ChatML), extract prompt and completion properly
                        messages = example["messages"]

                        # Extract user and assistant messages separately
                        user_messages = [msg for msg in messages if msg["role"] != "assistant"]
                        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                        if user_messages and assistant_messages:
                            
                            r1_suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."

                            # 把后缀拼接到原有的 prompt 后面
                            user_messages[-1]["content"] = user_messages[-1]["content"] + r1_suffix
                            
                            messages = user_messages + assistant_messages

                            prompt_text = processing_class.apply_chat_template(
                                user_messages,
                                add_generation_prompt=True,  # add assistant prompt
                                tokenize=False,
                                **example.get("chat_template_kwargs", {}),
                            )

                            # Get the full conversation with assistant response
                            full_text = processing_class.apply_chat_template(
                                messages,
                                add_generation_prompt=False,
                                tokenize=False,
                                **example.get("chat_template_kwargs", {}),
                            )

                            # Extract completion as everything after the prompt
                            # This ensures we capture any extra tokens (like <think> tags) that the template adds
                            if full_text.startswith(prompt_text):
                                completion_text = full_text[len(prompt_text) :]
                            else:
                                # Fallback: use assistant content + EOS
                                assistant_content = assistant_messages[0]["content"]
                                completion_text = (
                                    assistant_content + processing_class.eos_token
                                    if hasattr(processing_class, "eos_token")
                                    else assistant_content
                                )

                            # Store original text for cross-tokenizer distillation
                            result["original_prompt_text"] = prompt_text
                            result["original_completion_text"] = completion_text
                        else:
                            # Fallback: use empty prompt and full text as completion
                            full_text = processing_class.apply_chat_template(
                                messages, tokenize=False, **example.get("chat_template_kwargs", {})
                            )
                            result["original_prompt_text"] = ""
                            result["original_completion_text"] = full_text

                        # Process the conversation normally
                        processed = processing_class.apply_chat_template(
                            example["messages"],
                            return_dict=True,
                            return_assistant_tokens_mask=assistant_only_loss,
                            **example.get("chat_template_kwargs", {}),
                        )
                        if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                            raise RuntimeError(
                                "You're using `assistant_only_loss=True`, but at least one example has no "
                                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                "generate assistant masks — it may be missing the `{% generation %}` tag. Please "
                                "check the template and ensure it's correctly configured to support assistant "
                                "masking."
                            )
                        result.update({k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed})
                        # Add attention_mask if not already present
                        if "attention_mask" not in result:
                            result["attention_mask"] = [1] * len(result["input_ids"])
                    else:
                        # For regular language modeling, store the full text as completion and empty prompt
                        result["original_prompt_text"] = ""
                        result["original_completion_text"] = example.get(dataset_text_field, example.get("text", ""))

                        tokenized = processing_class(text=example[dataset_text_field])
                        result.update(
                            {
                                "input_ids": tokenized.input_ids,
                                "attention_mask": getattr(tokenized, "attention_mask", [1] * len(tokenized.input_ids)),
                            }
                        )

                return result

            dataset = dataset.map(
                tokenize_with_original_text,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "assistant_only_loss": args.assistant_only_loss,
                },
                **map_kwargs,
            )
            
            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns_to_keep = ["input_ids", "original_prompt_text", "original_completion_text"]
                existing_columns = set(dataset.column_names)
                columns_to_select = [col for col in columns_to_keep if col in existing_columns]

                dataset = dataset.select_columns(columns_to_select)
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

            if args.use_liger_kernel:
                required_columns = {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "completion_mask",
                    "assistant_masks",
                    "original_prompt_text",
                    "original_completion_text",
                }
                dataset = dataset.select_columns(required_columns.intersection(dataset.column_names))

        return dataset