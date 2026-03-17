import json

with open("/code/pruning_lrm_pipeline/Qwen2.5-Math/evaluation/data/deepscaler/train.jsonl", "r") as f:
    data = json.load(f)   # 注意：是 load，不是 loads

assert isinstance(data, list)

with open("deepscaler_conversation.jsonl", "w") as fout:
    for item in data:
        messages = [
            {
                "role": "user",
                "content": item["problem"]
            },
            {
                "role": "assistant",
                "content": item["solution"]
            }
        ]
        fout.write(
            json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
        )
