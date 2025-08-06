import json
import random


# === 設定參數 ===
input_file = "data/cot_map_11x11_311to313.jsonl"  # 原始資料檔
output_file = "data/cot_map_11x11.jsonl_311to313"  # 打亂後的新檔名

# === 讀取資料 ===
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# === 打亂順序 ===
random.shuffle(data)

# === 輸出檔案 ===
with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已將 {input_file} 打亂並儲存為 {output_file}，共 {len(data)} 筆")
