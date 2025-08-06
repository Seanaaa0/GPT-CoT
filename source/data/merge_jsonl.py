import os
import random
import json

# === 輸入檔案名稱 ===
input_files = [
    # "cot_10x10_1001to1200_99nlp.jsonl",
    # "cot_10x10_2001to2200_09nlp.jsonl",
    # "cot_10x10_3001to3200_90nlp.jsonl",
    "nlp_10x10_multi_entre_shuffled3.jsonl",
    "nlp_straight_andmore1.jsonl"
]

# === 合併並打散後儲存路徑 ===
output_file = "nlp_straight_andmore1.jsonl"

# === 讀取所有資料並合併 ===
all_data = []
for file_name in input_files:
    with open(file_name, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"❌ JSON 格式錯誤：{file_name} 中有壞行")
                continue

# === 打亂順序 ===
random.shuffle(all_data)

# === 輸出為新檔案 ===
with open(output_file, "w", encoding="utf-8") as fout:
    for item in all_data:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 打散並合併完成，共 {len(all_data)} 筆資料 → {output_file}")
