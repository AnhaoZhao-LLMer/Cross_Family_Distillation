from dataclasses import dataclass, field
from typing import Optional
from ..gold import GOLDConfig  # 请根据你的实际目录层级确认这个导入路径

@dataclass
class CrossFamilyDistillConfig(GOLDConfig):
    """
    Configuration class for CrossFamilyPolicyDistillTrainer.
    """
    
    # --- Cross-Family 特有参数 ---
    teacher_tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher 模型的 tokenizer 路径或名称。跨词表蒸馏必填。"}
    )
    
    # --- 核心蒸馏/RL 模式控制 (我们刚才新增的) ---
    distill_method: str = field(
        default="rl_based",
        metadata={"help": "使用的训练方法: 'seq_level_kd' (序列级逆向 KL) 或 'rl_based' (基于 GRPO 的强化学习)。"}
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "vLLM 生成的组大小 (G)。仅在 distill_method='rl_based' 时生效。"}
    )

    # --- 奖励与 Advantage 计算参数 ---
    reward_normalize: str = field(
        default="per_teacher_token",
        metadata={"help": "奖励归一化方式: none | per_teacher_token | per_student_token"}
    )
    normalize_advantage: bool = field(
        default=True, 
        metadata={"help": "是否在 rl_based 模式下对 Advantage 进行标准化 (除以标准差)"}
    )
    
    # --- Baseline 参数 (仅作备用或兼容旧版逻辑) ---
    baseline_type: str = field(
        default="ema",
        metadata={"help": "Baseline 类型: none | ema | batch_mean (当前 rl_based 默认使用组内均值)"}
    )
    baseline_momentum: float = field(
        default=0.9, 
        metadata={"help": "EMA baseline 的动量参数"}
    )
    
    # --- 策略与参考模型参数 ---
    use_old_policy: bool = field(
        default=True, 
        metadata={"help": "是否在 Rollout 时保存旧策略的 logprobs (类似 GRPO/PPO 的做法)。"}
    )
    beta_kl: float = field(
        default=0.0, 
        metadata={"help": "如果引入 KL 惩罚，对应的权重系数 (可选)。"}
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "参考模型的路径或名称，用于计算 KL 散度 (可选)。"}
    )

    # --- 多数据集评测采样参数 ---
    eval_test_names: str = field(
        default="gsm8k",
        metadata={"help": "数据集名称，多个用逗号分隔。"}
    )
    eval_test_paths: str = field(
        default="",
        metadata={"help": "数据集 JSONL 路径，多个用逗号分隔。"}
    )
    eval_test_samples: str = field(
        default="all",
        metadata={"help": "每个数据集评估的数量。'all' 表示全量，或用逗号分隔数字。例如: '100,200'"}
    )

    # --- 评估 Sampling 参数 ---
    eval_temperature: float = field(
        default=0.0,
        metadata={"help": "评估时的采样温度。0.0 为贪心解码（Greedy Decoding）。"}
    )
    eval_n: int = field(
        default=1,
        metadata={"help": "评估时每个 Prompt 生成的回答数量 (用于 Pass@K)。"}
    )
    eval_max_new_tokens: int = field(
        default=None,
        metadata={"help": "评估时的最大生成长度。如果为 None，则默认使用训练时的 max_completion_length。"}
    )

    # --- 评估频率控制 ---
    eval_steps: int = field(
        default=60,
        metadata={"help": "每隔多少步（Global Steps）执行一次自定义的 vLLM 测试集评估。"}
    )