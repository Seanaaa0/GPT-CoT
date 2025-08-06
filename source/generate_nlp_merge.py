import os
import json
import random
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

# === 可調整參數 ===
SEED_RANGES = {
    "16": (8001, 8100),
    "81": (2001, 2100),
    "67": (3001, 3100),
    # "97": (1001, 1050),
    # "40": (5001, 5050),
    # "69": (6001, 6050),
    # "82": (7001, 7050),
    # "17": (8001, 8050),
    # "23": (9001, 9050),
}

START_POSITIONS = {
    "16": (1, 6),
    "81": (8, 1),
    "67": (6, 7),
    # "97": (9, 7),
    # "40": (4, 0),
    # "69": (6, 9),
    # "82": (8, 2),
    # "17": (1, 7),
    # "23": (2, 3),

}

SIZE = 10
OUTPUT_DIR = "data"
MERGED_OUTPUT_FILE = "nlp_10x10__shuffled4.jsonl"

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
