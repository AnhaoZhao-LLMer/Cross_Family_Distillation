import time
from transformers import TrainerCallback
from datasets import load_dataset
from math_verify import parse, verify
from vllm import SamplingParams

class VLLMMathEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        args = trainer.args
        # 1. 从 trainer.args 中读取参数
        self.eval_steps = getattr(args, "eval_steps", 60)
        names_str = getattr(args, "eval_test_names", "")
        paths_str = getattr(args, "eval_test_paths", "")
        samples_str = getattr(args, "eval_test_samples", "all")
        
        # 如果没有传评估数据集，直接跳过初始化
        if not names_str or not paths_str:
            self.all_test_data = {}
            return

        self.names = [n.strip() for n in names_str.split(",")]
        self.paths = [p.strip() for p in paths_str.split(",")]
        
        sample_list = [s.strip() for s in samples_str.split(",")]
        if len(sample_list) == 1:
            self.sample_limits = [sample_list[0]] * len(self.names)
        else:
            self.sample_limits = sample_list

        if len(self.names) != len(self.paths) or len(self.names) != len(self.sample_limits):
            raise ValueError("数据集的 names, paths, samples 数量不匹配！")
            
        # 2. 提前加载数据
        self.all_test_data = {}
        # 为了避免多进程同时读文件卡死，建议只让 main_process 加载
        if self.trainer.accelerator.is_main_process:
            for name, path, limit in zip(self.names, self.paths, self.sample_limits):
                self.all_test_data[name] = self._prepare_dataset(name, path, limit)

    def _prepare_dataset(self, name, path, limit):
        raw_dataset = load_dataset("json", data_files=path, split="train")
        if limit.lower() != "all":
            try:
                max_n = min(len(raw_dataset), int(limit))
                # 推荐加个 seed 保证每次加载一致
                raw_dataset = raw_dataset.shuffle(seed=42).select(range(max_n))
            except ValueError:
                pass

        processed_data = []
        # 🌟 获取 Trainer 里的 Tokenizer
        tokenizer = self.trainer.processing_class
        
        for item in raw_dataset:
            # 1. 解析原始问题和答案
            if name.lower() == "gsm8k":
                raw_prompt = item["question"]
                gold = item["answer"].split("####")[-1].strip()
            elif name.lower() == "math":
                raw_prompt = item.get("problem", "")
                gold = item.get("solution", "")
            else:
                raw_prompt = item.get("question", item.get("prompt", ""))
                gold = item.get("answer", item.get("gold", ""))

            # 🌟 2. 构建标准的 Message 格式，并在末尾追加 R1 专属后缀
            r1_suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
            
            messages = [
                {"role": "user", "content": raw_prompt + r1_suffix}
            ]
            
            # 🌟 3. 应用 Chat Template (转为字符串格式)
            # add_generation_prompt=True 极其重要！它会自动帮你加上 "<|im_start|>assistant\n" 让模型知道该回答了
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 移除 DeepSeek 特有的起始符
            bos_token = tokenizer.bos_token  # 或者是 "<｜begin▁of▁sentence｜>"
            if formatted_prompt.startswith(bos_token):
                formatted_prompt = formatted_prompt[len(bos_token):]
                
            processed_data.append({
                "prompt": formatted_prompt, 
                "gold": gold
            })
        
        print(f"📦 Dataset '{name}': Loaded and formatted {len(processed_data)} samples.")
        return processed_data
    

    # def on_step_end(self, args, state, control, **kwargs):
    #     # 如果没有数据集，或者还没到 eval_steps，直接返回
    #     import ipdb; ipdb.set_trace()
    #     if not getattr(self, "all_test_data", None):
    #         return control
    #     if state.global_step % self.eval_steps != 0:
    #         return control
        
    #     if not getattr(self.trainer, "use_vllm", False):
    #         return control
    #     if not self.trainer.accelerator.is_main_process:
    #         return control

    #     eval_temp = getattr(self.trainer.args, "eval_temperature", 0.0)
    #     eval_n = getattr(self.trainer.args, "eval_n", 1)
    #     eval_max_tokens = getattr(self.trainer.args, "eval_max_new_tokens", getattr(self.trainer.args, "max_completion_length", 512))

    #     eval_sampling_params = SamplingParams(
    #         n=eval_n, 
    #         temperature=eval_temp, 
    #         max_tokens=eval_max_tokens
    #     )

    #     is_colocate = (self.trainer.vllm_mode == "colocate")
    #     if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
    #         self.trainer.vllm_engine.wake_up()

    #     for name, data in self.all_test_data.items():
    #         if not data: continue
            
    #         print(f"🚀 Eval {name} (N={len(data)}) | Temp={eval_temp} | n={eval_n} | Step {state.global_step}...")
    #         start_time = time.time()
            
    #         prompts = [item["prompt"] for item in data]
    #         golds = [item["gold"] for item in data]

    #         all_outputs = self.trainer.vllm_engine.generate(prompts, sampling_params=eval_sampling_params, use_tqdm=True)

    #         correct_count = 0
    #         total_len = 0
            
    #         for i, output in enumerate(all_outputs):
    #             gold_parsed = parse(golds[i])
    #             any_correct = False
                
    #             for comp in output.outputs:
    #                 pred_text = comp.text
    #                 total_len += len(comp.token_ids)
    #                 try:
    #                     is_correct = verify(gold_parsed, parse(pred_text))
    #                 except:
    #                     is_correct = False
                        
    #                 if is_correct:
    #                     any_correct = True
    #                     break 
                
    #             if any_correct:
    #                 correct_count += 1

    #         acc = correct_count / len(data)
    #         avg_len = total_len / (len(data) * eval_n)
            
    #         self.trainer.log({
    #             f"eval/{name}_acc": acc,
    #             f"eval/{name}_avg_len": avg_len,
    #             "step": state.global_step
    #         })
    #         print(f"📊 {name} Acc: {acc:.4f} | Avg Len: {avg_len:.1f}")

    #     if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
    #         self.trainer.vllm_engine.sleep(level=2)

    #     return control

    # ... 前面的 __init__ 和 _prepare_dataset 保持不变 ...

    def _do_eval(self, state):
        """🌟 核心封装：只做评估，不管什么时候被调用"""
        if not getattr(self.trainer, "use_vllm", False):
            return

        # ⚠️ 如果启用了多卡 Tensor Parallelism，vLLM 要求所有 rank 必须同时进入 generate，
        # 所以不能在这里简单地 return 掉非主进程，而是要让它们一起走下面的逻辑。
        is_main = self.trainer.accelerator.is_main_process

        # 🌟 1. 扩充并对齐 SamplingParams
        eval_temp = getattr(self.trainer.args, "eval_temperature", 0.0)
        eval_n = getattr(self.trainer.args, "eval_n", 1)
        eval_max_tokens = getattr(self.trainer.args, "eval_max_new_tokens", getattr(self.trainer.args, "max_completion_length", 512))
        
        # 从 args 中捞取更多 rollout 参数，没有 eval_ 前缀就 Fallback 到训练参数
        eval_top_p = getattr(self.trainer.args, "eval_top_p", getattr(self.trainer.args, "top_p", 1.0))
        eval_top_k = getattr(self.trainer.args, "eval_top_k", getattr(self.trainer.args, "top_k", -1))
        eval_min_p = getattr(self.trainer.args, "eval_min_p", getattr(self.trainer.args, "min_p", 0.0))
        eval_rep_penalty = getattr(self.trainer.args, "eval_repetition_penalty", getattr(self.trainer.args, "repetition_penalty", 1.0))

        eval_sampling_params = SamplingParams(
            n=eval_n, 
            temperature=eval_temp, 
            top_p=eval_top_p,
            top_k=eval_top_k,
            # min_p=eval_min_p,
            # repetition_penalty=eval_rep_penalty,
            max_tokens=eval_max_tokens
        )

        is_colocate = (self.trainer.vllm_mode == "colocate")
        if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
            self.trainer.vllm_engine.wake_up()

        # 确保只有主进程有日志输出
        for name, data in self.all_test_data.items():
            if not data: continue
            
            if is_main:
                print(f"🚀 Eval {name} (N={len(data)}) | Temp={eval_temp} | RepPen={eval_rep_penalty} | Step {state.global_step}...")
                start_time = time.time()
            
            prompts = [item["prompt"] for item in data]
            golds = [item["gold"] for item in data]

            # 🌟 2. 完美模仿训练时的 Tensor Parallelism 多卡分发逻辑
            if is_colocate and hasattr(self.trainer, "vllm_tp_group") and self.trainer.vllm_tensor_parallel_size > 1:
                import torch
                orig_size = len(prompts)
                gathered_prompts = [None for _ in range(self.trainer.vllm_tensor_parallel_size)]
                # 将 prompt 分发到各个进程
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.trainer.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts

            # 执行多卡同步推理
            all_outputs = self.trainer.vllm_engine.generate(all_prompts_text, sampling_params=eval_sampling_params, use_tqdm=is_main)
    

            # 如果只有主进程需要统计 Acc
            if is_main:
                # 把 TP 模式下生成的结果再切分回原来的大小
                if is_colocate and hasattr(self.trainer, "vllm_tp_group") and self.trainer.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.trainer.vllm_tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size * eval_n, (local_rank_in_group + 1) * orig_size * eval_n)
                    # 从 all_outputs 中抽取出当前批次对应的那部分
                    outputs_to_eval = all_outputs[tp_slice]
                else:
                    outputs_to_eval = all_outputs

                correct_count = 0
                total_len = 0
                
                for i, output in enumerate(outputs_to_eval):
                    gold_parsed = parse(golds[i])
                    any_correct = False
                    
                    for comp in output.outputs:
                        pred_text = comp.text
                        total_len += len(comp.token_ids)
                        try:
                            is_correct = verify(gold_parsed, parse(pred_text))
                        except:
                            is_correct = False
                            
                        if is_correct:
                            any_correct = True
                            break 
                    
                    if any_correct:
                        correct_count += 1
                
                acc = correct_count / len(data)
                avg_len = total_len / (len(data) * eval_n)
                
                self.trainer.log({
                    f"eval/{name}_acc": acc,
                    f"eval/{name}_avg_len": avg_len,
                    "step": state.global_step
                })
                print(f"📊 {name} Acc: {acc:.4f} | Avg Len: {avg_len:.1f} | Time: {time.time()-start_time:.1f}s")

        if is_colocate and getattr(self.trainer, "vllm_enable_sleep_mode", False):
            self.trainer.vllm_engine.sleep(level=2)

    def on_step_begin(self, args, state, control, **kwargs):
        """🌟 新增：在一切训练开始前打个 Baseline"""
        if not getattr(self, "all_test_data", None):
            return control
        
        # 使用标记位确保只在第一步执行一次初始评估
        if not getattr(self, "_initial_eval_done", False):
            print(f"\n📊 [Baseline] Running initial evaluation before parameter updates...")
            self._do_eval(state)
            self._initial_eval_done = True # 标记为已完成
            
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """🌟 简化：正常的周期性评估"""
       
        if not getattr(self, "all_test_data", None):
            return control
            
        # 每隔 eval_steps 调用一次封装好的函数
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._do_eval(state)
            
        return control