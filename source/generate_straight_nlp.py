import os
import json
import random

# === 參數 ===
SIZE = 10
OUTPUT_PATH = "data/straight_line_nlp.jsonl"
INSTRUCTION_POOL = "Based on the directional steps, determine the final position with reasoning."
INSTRUCTION = random.choice(INSTRUCTION_POOL)
# 起點與最多幾步
START_POINTS = [(8, 0), (5, 5), (3, 7), (0, 5), (6, 4)]
STEPS_PER_DIRECTION = 25  # 每個方向生成幾筆資料

DIRECTION_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}


def generate_record(start, direction, steps):
    x, y = start
    action_list = []
    trace = [f"Start at ({x},{y})"]
    for i in range(steps):
        dx, dy = DIRECTION_VEC[direction]
        nx, ny = x + dx, y + dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE:
            x, y = nx, ny
            action_list.append(direction)
            trace.append(f"Step {i+1}: move {direction} → ({x},{y})")
        else:
            break
    trace.append(f"Final position: ({x},{y})")

    return {
        "instruction": INSTRUCTION,
        "input": f"Start at {start}\nActions: {', '.join(action_list)}",
        "output": "\n".join(trace)
    }


def generate_all():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_data = []

    for start in START_POINTS:
        for dir in DIRECTION_VEC.keys():
            for _ in range(STEPS_PER_DIRECTION):
                steps = random.randint(1, 6)
                record = generate_record(start, dir, steps)
                all_data.append(record)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 共產出 {len(all_data)} 筆直線 NLP 訓練資料 → {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_all()
