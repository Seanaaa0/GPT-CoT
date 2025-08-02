from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from flask_cors import CORS
import re

base_model_path = "/home/seana/axolotl_project/models/phi-2/phi2"
lora_checkpoint_path = "/home/seana/axolotl_project/outputs/phi2-NLP-finetune2/checkpoint-1480"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型與 tokenizer 載入 ===
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model = PeftModel.from_pretrained(model, lora_checkpoint_path).to(device)
model.eval()

app = Flask(__name__)
CORS(app)


def run_model(start, actions):
    x, y = start
    action_str = f"Start at ({x},{y})\nActions: " + ", ".join(actions)

    print("=== GPT 接收輸入 ===")
    print(action_str)

    prompt = (
        "You are navigating a 10x10 grid environment. Starting from a given position, you are provided with a sequence of directional instructions in natural language (e.g., right, left, up, down).\n"
        "Follow each instruction step by step, updating your position accordingly. For each step, show the new position and clearly explain the movement.\n"
        "Use the following format:\n\n"
        "Start at (x, y)\n"
        "Step 1: move <direction> → (new_x1, new_y1)\n"
        "...\n"
        "Final position: (x_final, y_final)\n\n"
        f"{action_str}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== GPT 輸出回應 ===")
    print(result)

    return result


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    start = data.get("start", [0, 0])
    actions = data.get("actions", [])
    print("=== 收到 POST /inference 請求，起點為：", start)
    print("=== 動作序列：", actions)

    gpt_response = run_model(start, actions)

    return jsonify({
        "gpt_output": gpt_response
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
