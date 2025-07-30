from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import re
import time

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-CoT-finetune/checkpoint-336"
test_data_path = "/home/seana/axolotl_project/data/cot_10x10_1001to1050.jsonl"
max_tokens = 8  # 小輸出限制，加快速度

# === 載入模型（強制 CPU + float32，避免卡住） ===
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

# === 強化版：從完整輸出中抽出 Final position (x,y)，容許負號 ===


def extract_final_position(text):
    match = re.search(r"Final position:\s*\((\-?\d+),\s*(\-?\d+)\)", text)
    return match.group(0) if match else "[INVALID]"


# === 推論主程式 ===
total = 0
correct = 0

with open(test_data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break  # 先跑前 30 筆
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        expected = data["output"].strip()

        # --- 新版 prefix（向量形式）
        prefix = (
            "You are in a 10x10 grid. You always start at position (0,0).\n"
            "Each action is represented as a 2D vector (dx, dy).\n"
            "Apply each (dx, dy) in order, one step at a time.\n"
            "Return the final position (x, y) after all steps.\n\n"
        )
        prompt = prefix + \
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

        start = time.time()

        with torch.no_grad():  # ✅ 加速推論
            output_raw = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=max_tokens,
                do_sample=False
            )

        decoded = tokenizer.decode(output_raw[0], skip_special_tokens=True)
        predicted = extract_final_position(decoded.replace(prompt, "").strip())
        end = time.time()

        is_correct = predicted == expected
        correct += is_correct
        total += 1

        print(f"[{i+1}]  Correct: {is_correct} | 🕒 {end - start:.2f} sec")
        print(f"    → Output:   {predicted}")
        print(f"    → Expected: {expected}")

        # ✅ 顯示錯誤樣本與 prompt
        if not is_correct:
            print(f"❌ Mismatch! Prompt:\n{prompt}\n")

print(f"\n✅ Accuracy: {correct}/{total} = {correct / total:.2%}")
