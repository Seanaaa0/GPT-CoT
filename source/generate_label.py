import os
import json
import random
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

# === 可調整參數 ===
SEED_RANGES = {

    "86": (4001, 4100),
    "93": (5001, 5100),
    "07": (6001, 6100),
}

START_POSITIONS = {

    "86": (8, 6),
    "93": (9, 3),
    "07": (0, 7),
}

SIZE = 10
OUTPUT_DIR = "data"
MERGED_OUTPUT_FILE = "10x10_test_labeled1.jsonl"

instruction_pool = [
    "Given a sequence of actions, infer the final position from the starting point with detailed steps.",
    "You are navigating a grid. Start at the given position and execute the actions one by one. Where will you end up?"
]


def generate_cot_and_label(start, goal, actions, grid_size):
    x, y = start
    visited = set()
    visited.add(start)
    steps = [f"Start at {start}"]
    out_of_bound = False
    loop_detected = False

    for i, (dx, dy) in enumerate(actions):
        x += dx
        y += dy
        steps.append(f"Step {i+1}: move ({dx},{dy}) → ({x},{y})")

        if not (0 <= x < grid_size and 0 <= y < grid_size):
            out_of_bound = True
        elif (x, y) in visited:
            loop_detected = True
        else:
            visited.add((x, y))

    steps.append(f"Final position: ({x},{y})")

    # 判斷 label 類型
    if out_of_bound:
        label = "out of bound"
    elif (x, y) == goal:
        label = "correct"
    elif (x, y) != goal:
        dx = x - goal[0]
        dy = y - goal[1]
        if (abs(dx) + abs(dy)) < len(actions):
            label = "too long"
        elif (abs(dx) + abs(dy)) > len(actions):
            label = "too short"
        elif loop_detected:
            label = "loop"
        else:
            label = "wrong"
    else:
        label = "unknown"

    return "\n".join(steps), label


def generate_records():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    for tag, (seed_start, seed_end) in SEED_RANGES.items():
        start_pos = START_POSITIONS[tag]
        output_path = os.path.join(
            OUTPUT_DIR, f"vec_cot_{SIZE}x{SIZE}_{seed_start}to{seed_end}_{tag}_labeled.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for seed in range(seed_start, seed_end + 1):
                env = SimpleEnv(size=SIZE, seed=seed)
                env.start = start_pos
                goal = env.get_goal()
                full_actions = simple_greedy_policy(start_pos, goal)
                actions = [eval(a["vec"])
                           for a in full_actions]  # [(0,1), (1,0), ...]

                input_text = f"Start={start_pos}\nActions: " + \
                    ", ".join([str(a) for a in actions])
                output_text, label = generate_cot_and_label(
                    start_pos, goal, actions, SIZE)

                record = {
                    "instruction": random.choice(instruction_pool),
                    "input": input_text,
                    "output": output_text,
                    "label": label
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_data.append(record)

        print(f"✅ 完成 {tag} 訓練資料：{output_path} ({seed_end - seed_start + 1} 筆)")

    return all_data


def merge_and_shuffle(all_data):
    random.shuffle(all_data)
    merged_path = os.path.join(OUTPUT_DIR, MERGED_OUTPUT_FILE)
    with open(merged_path, "w", encoding="utf-8") as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 所有資料已合併並打散，共 {len(all_data)} 筆 → {merged_path}")


if __name__ == "__main__":
    all_records = generate_records()
    merge_and_shuffle(all_records)
