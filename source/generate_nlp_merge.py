import os
import json
import random
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

# === 可調整參數 ===
SEED_RANGES = {
    "00": (14001, 14010),
    "09": (12001, 12010),
    "90": (13001, 13010),
    "99": (11001, 11010),
    "55": (15001, 15010),
}

START_POSITIONS = {
    "00": (0, 0),
    "09": (0, 9),
    "90": (9, 0),
    "99": (9, 9),
    "55": (5, 5),
}

SIZE = 10
OUTPUT_DIR = "data"
MERGED_OUTPUT_FILE = "nlp_10x10_multi_entre_shuffled2.jsonl"

instruction_text = "Given a sequence of actions like 'right, down', infer the final position starting from (5,5) with clear reasoning."


def generate_cot_steps(start, actions):
    x, y = start
    steps = [f"Start at ({x},{y})"]
    for i, act in enumerate(actions):
        dx, dy = eval(act["vec"])
        x += dx
        y += dy
        steps.append(
            f"Step {i+1}: move {act['word']} ({act['vec']}) → ({x},{y})")
    steps.append(f"Final position: ({x},{y})")
    return "\n".join(steps)


def generate_records():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    for tag, (seed_start, seed_end) in SEED_RANGES.items():
        start_pos = START_POSITIONS[tag]
        output_path = os.path.join(
            OUTPUT_DIR, f"cot_{SIZE}x{SIZE}_{seed_start}to{seed_end}_{tag}nlp.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for seed in range(seed_start, seed_end + 1):
                env = SimpleEnv(size=SIZE, seed=seed)
                env.start = start_pos
                goal = env.get_goal()
                actions = simple_greedy_policy(start_pos, goal)
                instruction = random.choice(instruction_text)
                words = [a["word"] for a in actions]
                input_text = f"Start={start_pos}\nActions: " + ", ".join(words)
                output_text = generate_cot_steps(start_pos, actions)

                record = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
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
