import argparse
import json
import os
import random
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

# === CoT 風格步驟轉換 ===


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

# === CLI 程式：生成 CoT 格式資料 ===


def generate_cot_dataset(seed_start=1, seed_end=100, size=5, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"cot_{size}x{size}_{seed_start}to{seed_end}_00nlp.jsonl"
    output_path = os.path.join(output_dir, output_name)

    instruction_text = "Given a sequence of actions like 'right, down', infer the final position starting from (5,5) with clear reasoning."

    with open(output_path, "w", encoding="utf-8") as f:
        for seed in range(seed_start, seed_end + 1):
            start = (0, 0)
            env = SimpleEnv(size=size, seed=seed)
            env.start = start
            goal = env.get_goal()
            actions = simple_greedy_policy(start, goal)
            instruction = random.choice(instruction_text)

            words = [a["word"] for a in actions]
            input_text = f"Start={start}\nActions: " + ", ".join(words)
            output_text = generate_cot_steps(start, actions)

            record = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=10000)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    path = generate_cot_dataset(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        size=args.size,
        output_dir=args.output_dir
    )

    print(f"✅ CoT-style dataset saved to: {path}")
