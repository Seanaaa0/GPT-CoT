import argparse
import os
import json
import random
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

START_POSITIONS = [
    (4, 4), (4, 5), (4, 6),
    (5, 4), (5, 5), (5, 6),
    (6, 4), (6, 5), (6, 6),
    (0, 0), (10, 10)
]


def generate_cot_steps(start, actions):
    x, y = start
    steps = [f"Start at ({x},{y})"]
    for i, (dx, dy) in enumerate(actions):
        x += dx
        y += dy
        steps.append(f"Step {i+1}: move ({dx},{dy}) → ({x},{y})")
    steps.append(f"Final position: ({x},{y})")
    return "\n".join(steps)


def generate_cot_dataset(seed_start, seed_end, size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"cot_{size}x{size}_{seed_start}to{seed_end}_1.jsonl"
    output_path = os.path.join(output_dir, output_name)

    instruction_text = f"Given a list of (dx, dy) actions starting from (x,y), calculate the final position step-by-step and explain each step clearly."

    with open(output_path, "w", encoding="utf-8") as f:
        for start in START_POSITIONS:
            for seed in range(seed_start, seed_start + 10):
                env = SimpleEnv(size=size, seed=seed)
                goal = env.get_goal()
                path = simple_greedy_policy(start, goal)
                actions = [eval(p["vec"]) for p in path]

                record = {
                    "instruction": instruction_text,
                    "input": f"Start=({start[0]},{start[1]})\nActions: " + ", ".join([str(a) for a in actions]),
                    "output": generate_cot_steps(start, actions)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ CoT-style dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=100)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    generate_cot_dataset(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        size=args.size,
        output_dir=args.output_dir
    )
