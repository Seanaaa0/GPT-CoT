import json
import random

# === 設定參數 ===
input_file = "cot_10x10_multi_entre_shuffled3.jsonl"  # 原始資料檔
output_file = "random_multi_3.jsonl"                    # 輸出檔案名稱
num_samples = 50                                   # 要抽取的樣本數量

# === 讀取資料 ===
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# === 檢查數量是否足夠 ===
if num_samples > len(data):
    raise ValueError(f"資料不足，檔案僅有 {len(data)} 筆樣本")

# === 隨機抽樣 ===
sampled = random.sample(data, num_samples)

# === 輸出檔案 ===
with open(output_file, "w", encoding="utf-8") as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 成功從 {input_file} 中抽取 {num_samples} 筆樣本，儲存為 {output_file}")
