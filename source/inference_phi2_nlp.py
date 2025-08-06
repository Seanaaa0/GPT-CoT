from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time
import os

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-NLP-finetune1/checkpoint-952"
test_data_path = "/home/seana/axolotl_project/data/nlp_10x10__shuffled4.jsonl"
output_save_path = "/home/seana/axolotl_project/source/test_output/test_nlp4.jsonl"
max_tokens = 512

# === 強制使用 CPU ===
device = "cpu"
torch_dtype = torch.float32

# === 載入 tokenizer 與模型（用 CPU）===
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch_dtype,
    device_map={"": "cpu"}
).to(device)

model = PeftModel.from_pretrained(
    model,
    lora_checkpoint_path,
    device_map={"": "cpu"}
).to(device)

model.eval()

# === 回傳推論內容 ===


def extract_full_trace(text, prompt):
    return text.replace(prompt, "").strip()


# === 推論主程式 ===
total = 0
correct = 0
results = []

os.makedirs(os.path.dirname(output_save_path), exist_ok=True)

with open(test_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        expected = data["output"].strip()

        prefix = (
            "You are navigating a 10x10 grid environment. Starting from a given position, you are provided with a sequence of directional instructions in natural language (e.g., right, left, up, down).\n"
            "Follow each instruction step by step, updating your position accordingly. For each step, show the new position and clearly explain the movement.\n"
            "Use the following format:\n"
            "Start at (x,y)\n"
            "Step 1: move <direction> → (new_x1, new_y1)\n"
            "Step 2: move <direction> → (new_x2, new_y2)\n"
            "...\n"
            "Final position: (x,y)\n\n"
            "Be careful to base each step on the result of the previous one.\n"
        )

        prompt = prefix + \
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

        start = time.time()

        with torch.no_grad():
            output_raw = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(device),
                max_new_tokens=max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        decoded = tokenizer.decode(output_raw[0], skip_special_tokens=True)
        full_trace = extract_full_trace(decoded, prompt)
        end = time.time()

        # === 抓模型生成的預測位置 ===
        match = re.search(
            r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", full_trace)
        if match:
            predicted = f"({match.group(1)},{match.group(2)})"
        else:
            last_add = re.findall(
                r"Step \d+: \((\-?\d+),\s*(\-?\d+)\) \+ \((\-?\d+),\s*(\-?\d+)\)", full_trace)
            if last_add:
                x, y, dx, dy = map(int, last_add[-1])
                predicted = f"({x + dx},{y + dy})"
            else:
                predicted = "[INVALID]"

        match = re.search(
            r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", expected)
        if match:
            expected_final = f"({match.group(1)},{match.group(2)})"
        else:
            fallback = re.findall(
                r"Step \d+: .*→\s*\((\-?\d+),\s*(\-?\d+)\)", expected)
            expected_final = f"({fallback[-1][0]},{fallback[-1][1]})" if fallback else "[INVALID]"

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
