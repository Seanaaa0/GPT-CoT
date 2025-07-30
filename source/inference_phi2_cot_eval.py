from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time

# === 設定區 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-cot-finetune/checkpoint-336"
test_data_path = "/home/seana/axolotl_project/data/cot_10x10_1001to1050.jsonl"
max_tokens = 16

# === 載入模型與 LORA 微調參數 ===
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(
    model, lora_checkpoint_path, is_trainable=False, device_map="auto", is_local=True)
model.eval()

# === 抽出第一個座標 (x,y) ===


def extract_first_coord(text):
    match = re.search(r"\((\-?\d+),\s*(\-?\d+)\)", text)
    return match.group(0) if match else "[INVALID]"


# === 開始推論 ===
total = 0
correct = 0

with open(test_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        expected = data["output"].strip()

        prompt = f"{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            output_raw = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        end = time.time()

        decoded = tokenizer.decode(output_raw[0], skip_special_tokens=True)
        predicted = extract_first_coord(decoded.replace(prompt, "").strip())

        is_correct = predicted == expected
        correct += is_correct
        total += 1

        print(f"[{i+1}]  Correct: {is_correct} | 🕒 {end - start:.2f} sec")
        print(f"    → Output:   {predicted}")
        print(f"    → Expected: {expected}")
        if not is_correct:
            print(f"❌ Mismatch! Prompt:\n{prompt}\n")

print(f"\n✅ Accuracy: {correct}/{total} = {correct / total:.2%}")
