from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

# === 設定 ===
base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-finetune2/checkpoint-213"
test_data_path = "/home/seana/axolotl_project/data/simple_10x10_601to1500.jsonl"

max_tokens = 32

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, lora_checkpoint_path)
model.eval()


def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# === 評估前 100 筆樣本的正確率 ===
# total = 0
# correct = 0

# with open(test_data_path, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         if i >= 100:  # 前 100 筆即可
#             break
#         data = json.loads(line)
#         instruction = data["instruction"]
#         input_text = data.get("input", "")
#         expected = data["output"].strip()

#         prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
#         output = generate_response(prompt).replace(prompt, "").strip()

#         is_correct = (output == expected)
#         correct += is_correct
#         total += 1

#         print(f"\n=== Sample #{i+1} ===")
#         print(f"✅ Correct: {is_correct}")
#         print(f"Prompt: {prompt}")
#         print(f"Expected: {expected}")
#         print(f"Predicted: {output}")

# print(
#     f"\n✅ Accuracy on 601–1500 set (前100): {correct}/{total} = {correct / total:.2%}")
# 程式上面是 tokenizer、model、generate_response 定義...

# === 你加在最底下這一段 ===
if __name__ == "__main__":
    prompt = "### Instruction:\nMove from (0,0) to (9,9) on a 10x10 grid.\n\n### Response:\n"
    print(generate_response(prompt))
