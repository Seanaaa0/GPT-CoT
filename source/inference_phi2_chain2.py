
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time
import os

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-CoT-finetune6/checkpoint-24"
test_data_path = "/home/seana/axolotl_project/data/random_multi_3.jsonl"
output_save_path = "/home/seana/axolotl_project/source/test_output/multi_test_3.jsonl"
max_tokens = 512
# 增加輸出長度以支援 chain-of-thought

# === 載入 tokenizer ===
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


def extract_full_trace(text, prompt):
    return text.replace(prompt, "").strip()


# === 推論主程式 ===
total = 0
correct = 0
results = []

os.makedirs(os.path.dirname(output_save_path), exist_ok=True)

with open(test_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        expected = data["output"].strip()

        prefix = (
            "You are in a 10x10 grid. The start position and a list of movement vectors are provided.\n"
            "Apply each vector one by one and show the result of each step in the following format:\n"
            "Start at (x,y)\n"
            "Step 1: (x1,y1) + (dx1,dy1) = (x2,y2)\n"
            "Step 2: (x2,y2) + (dx2,dy2) = (x3,y3)\n"
            "...\n"
            "Be careful to compute each step correctly based on the previous result.\n"
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
        full_trace = extract_full_trace(decoded, prompt)
        end = time.time()

        # === 抓模型生成的預測位置 ===
        # 優先從 "Final position" 擷取
        match = re.search(
            r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", full_trace)
        if match:
            predicted = f"({match.group(1)},{match.group(2)})"
        else:
            # 沒有 Final position 就抓最後一個 Step 的加總算出位置
            last_add = re.findall(
                r"Step \d+: \((\-?\d+),\s*(\-?\d+)\) \+ \((\-?\d+),\s*(\-?\d+)\)", full_trace)
            if last_add:
                x, y, dx, dy = map(int, last_add[-1])
                predicted = f"({x + dx},{y + dy})"
            else:
                predicted = "[INVALID]"

        # === 抓 Ground Truth ===
        match = re.search(
            r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", expected)
        if match:
            expected_final = f"({match.group(1)},{match.group(2)})"
        else:
            fallback = re.findall(
                r"Step \d+: .*→\s*\((\-?\d+),\s*(\-?\d+)\)", expected)
            expected_final = f"({fallback[-1][0]},{fallback[-1][1]})" if fallback else "[INVALID]"

        # === 比對結果 ===
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
            "output": full_trace
        })

# 儲存 jsonl
with open(output_save_path, "w", encoding="utf-8") as fout:
    for item in results:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Accuracy: {correct}/{total} = {correct / total:.2%}")
print(f"📁 Results saved to: {output_save_path}")
