import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# === 設定 ===
INPUT_PATH = "test_label2.jsonl"          # JSONL 路徑
OUTPUT_DIR = "trace_images2"              # 小圖輸出資料夾
GRID_SIZE = 10                           # 格子地圖大小
COMBINED_PREFIX = "trace_combined_"      # 合併圖檔前綴
COLS = 5                                 # 合併圖每行幾張
ROWS = 5                                 # 合併圖每列幾張
MAX_PER_PAGE = COLS * ROWS               # 每張合併圖最大圖數

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_color(label, correct):
    if correct:
        return "green"
    if label == "too short":
        return "orange"
    if label == "too long":
        return "blue"
    if label == "out of bound":
        return "black"
    if label == "loop":
        return "purple"
    return "red"


def parse_trace(text):
    steps = []
    lines = text.splitlines()
    start = final = None
    for line in lines:
        if line.startswith("Start at"):
            start = tuple(
                map(int, line.split("(")[1].split(")")[0].split(",")))
        elif line.startswith("Step"):
            parts = line.split("→")
            if len(parts) == 2:
                coord = parts[1].strip().strip("()")
                x, y = map(int, coord.split(","))
                steps.append((x, y))
        elif line.startswith("Final position:"):
            final = tuple(
                map(int, line.split(":")[1].strip().strip("()").split(",")))
    return start, steps, final


def draw_trace(idx, start, steps, final, label, correct):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE+1))
    ax.set_yticks(range(GRID_SIZE+1))
    ax.grid(True)

    ax.add_patch(patches.Rectangle(
        start, 1, 1, facecolor='yellow', edgecolor='black'))
    ax.text(start[0]+0.5, start[1]+0.5, "S",
            ha="center", va="center", fontsize=12)

    for i, (x, y) in enumerate(steps):
        ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor='lightgray', edgecolor='gray'))
        ax.text(x+0.5, y+0.5, str(i+1), ha="center", va="center", fontsize=8)

    fx, fy = final
    ax.add_patch(patches.Rectangle((fx, fy), 1, 1,
                 facecolor=get_color(label, correct), edgecolor='black'))
    ax.text(fx+0.5, fy+0.5, "F", ha="center",
            va="center", fontsize=12, color="white")

    ax.set_title(f"#{idx+1} - {'✅' if correct else '❌'} {label}")
    ax.invert_yaxis()
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/trace_{idx+1:02d}.png"
    plt.savefig(out_path)
    plt.close()


# === 主程式：產出小圖 ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        label = data.get("label", "unknown")
        correct = data.get("correct", False)
        output_text = data["output"]
        start, steps, final = parse_trace(output_text)
        draw_trace(i, start, steps, final, label, correct)

print(f"✅ Trace images generated in '{OUTPUT_DIR}'")

# === 合併小圖為大圖（每張最多 25 張）===
images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
num_pages = math.ceil(len(images) / MAX_PER_PAGE)

for page in range(num_pages):
    page_images = images[page * MAX_PER_PAGE:(page + 1) * MAX_PER_PAGE]
    loaded = [Image.open(os.path.join(OUTPUT_DIR, img)) for img in page_images]
    img_w, img_h = loaded[0].size

    combined = Image.new("RGB", (COLS * img_w, ROWS * img_h), color="white")
    for idx, img in enumerate(loaded):
        row = idx // COLS
        col = idx % COLS
        combined.paste(img, (col * img_w, row * img_h))

    combined.save(f"{COMBINED_PREFIX}{page+1}.png")
    print(f"📸 Saved combined image: {COMBINED_PREFIX}{page+1}.png")

print("✅ All trace visualization completed.")
