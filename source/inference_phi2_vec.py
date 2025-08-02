from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time
import os

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-Label-finetune1/checkpoint-150"
test_data_path = "/home/seana/axolotl_project/source/data/10x10_test_labeled1.jsonl"
output_save_path = "/home/seana/axolotl_project/source/data/test_output/test_label.jsonl"
max_tokens = 512

device = "cpu"
torch_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch_dtype,
    device_map={"": device}
).to(device)
model = PeftModel.from_pretrained(
    model, lora_checkpoint_path, device_map={"": device}).to(device)
model.eval()


def extract_full_trace(text, prompt):
    return text.replace(prompt, "").strip()


total = 0
correct = 0
results = []

os.makedirs(os.path.dirname(output_save_path), exist_ok=True)

with open(test_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        expected = data["output"].strip()
        label = data.get("label", "unknown")

        prefix = (
            "You are in a 10x10 grid. Starting from a given position and a sequence of (dx, dy) actions,\n"
            "compute the new position after each move, and return the final position.\n"
            "Follow this format:\n"
            "Start at (x,y)\n"
            "Step 1: move (dx,dy) → (new_x1, new_y1)\n"
            "...\n"
            "Final position: (x,y)\n\n"
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

        match = re.search(
            r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", full_trace)
        if match:
            predicted = f"({match.group(1)},{match.group(2)})"
        else:
            last_add = re.findall(
                r"Step \d+: .*→\s*\((\-?\d+),\s*(\-?\d+)\)", full_trace)
            predicted = f"({last_add[-1][0]},{last_add[-1][1]})" if last_add else "[INVALID]"

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
        print(f"    → Label    : {label}")
        print(f"    → Predicted: {predicted}")
        print(f"    → Expected : {expected_final}")
        print(f"    → Full Trace:\n{full_trace}\n")

        results.append({
            "instruction": instruction,
            "input": input_text,
            "output": full_trace,
            "label": label,
            "correct": is_correct
        })

with open(output_save_path, "w", encoding="utf-8") as fout:
    for item in results:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Accuracy: {correct}/{total} = {correct / total:.2%}")
print(f"📁 Results saved to: {output_save_path}")
