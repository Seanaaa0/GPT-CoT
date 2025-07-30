from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time
import os
from tqdm import tqdm

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-CoT-finetune/checkpoint-336"
test_data_path = "/home/seana/axolotl_project/data/cot_10x10_1051to1100.jsonl"
output_save_path = "/home/seana/axolotl_project/source/test_output/phi2_debug_output.jsonl"
max_tokens = 256
max_samples = 10  # 只跑前 N 筆作為 debug 用

# === 載入模型（使用 CPU）===
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(
    model, lora_checkpoint_path, device_map={"": "cpu"})
model.eval()

# === 推論主程式 ===
total = 0
correct = 0
results = []

os.makedirs(os.path.dirname(output_save_path), exist_ok=True)

with open(test_data_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f][:max_samples]

for i, data in enumerate(tqdm(lines, desc="Running debug samples")):
    instruction = data["instruction"]
    input_text = data["input"]
    expected = data["output"].strip()

    prefix = (
        "You are in a 10x10 grid. Start at position (0,0).\n"
        "You are given a list of 2D vectors (dx, dy).\n"
        "Please apply them one by one, and explain each step like this:\n"
        "Start at (0,0)\n"
        "Step 1: (0,0) + (+1,0) = (1,0)\n"
        "Step 2: (1,0) + (+1,0) = (2,0)\n"
        "...\n"
        "Final position: (x,y)\n\n"
    )

    prompt = prefix + \
        f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    start = time.time()
    with torch.no_grad():
        output_raw = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    decoded = tokenizer.decode(output_raw[0], skip_special_tokens=True)
    full_trace = decoded.replace(prompt, "").strip()
    end = time.time()

    # 預測：最後一個 step 的結果
    steps = re.findall(r"Step \d+: .*=\s*\((\-?\d+),\s*(\-?\d+)\)", full_trace)
    predicted = f"({steps[-1][0]},{steps[-1][1]})" if steps else "[INVALID]"

    # 正確答案：從 ground truth trace 中取出最後一個 step
    gt = re.findall(r"Step \d+: .*=\s*\((\-?\d+),\s*(\-?\d+)\)", expected)
    expected_final = f"({gt[-1][0]},{gt[-1][1]})" if gt else "[INVALID]"

    is_correct = predicted == expected_final
    correct += is_correct
    total += 1

    print(f"[{i+1}]  Correct: {is_correct} | 🕒 {end - start:.2f} sec")
    print(f"    → Predicted: {predicted}")
    print(f"    → Expected : {expected_final}")
    print(f"    → Full Trace:\n{full_trace}\n")

    results.append({
        "instruction": instruction,
        "input": input_text,
        "predicted": predicted,
        "expected": expected_final,
        "correct": is_correct,
        "output": full_trace
    })

# 儲存 log
with open(output_save_path, "w", encoding="utf-8") as fout:
    for item in results:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Accuracy: {correct}/{total} = {correct / total:.2%}")
print(f"📁 Results saved to: {output_save_path}")
