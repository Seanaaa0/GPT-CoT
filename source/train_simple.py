import argparse
import json
import os
import random  # ✅ 加這行
from maze_simple import SimpleEnv


def simple_greedy_policy(start, goal):
    path = []
    x, y = start
    gx, gy = goal

    while x != gx:
        if gx > x:
            x += 1
            path.append("(+1,0)")
        else:
            x -= 1
            path.append("(-1,0)")

    while y != gy:
        if gy > y:
            y += 1
            path.append("(0,+1)")
        else:
            y -= 1
            path.append("(0,-1)")

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=100)
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    output_name = f"simple_{args.size}x{args.size}_{args.seed_start}to{args.seed_end}.jsonl"
    output_path = os.path.join("data", output_name)

    # ✅ 加入隨機 instruction pool
    instruction_pool = [
        "You are at (0,0). Each action is a movement vector in the form (dx, dy). What is your final position?",
        "Starting from position (0,0), apply the following vectors and predict the resulting coordinates.",
        "Given a list of (dx, dy) actions starting from (0,0), calculate the final position.",
        "In a 10x10 grid, you move from (0,0) using these directional vectors. Where do you end up?",
        "Each step is represented as (dx, dy). Starting at (0,0), compute the final location.",
        "An agent starts at (0,0) and takes the following vector steps. What is the resulting position?"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for seed in range(args.seed_start, args.seed_end + 1):
            env = SimpleEnv(size=args.size, seed=seed)
            start = env.reset()
            goal = env.get_goal()
            actions = simple_greedy_policy(start, goal)

            instruction = random.choice(instruction_pool)  # ✅ 隨機抽一個

            record = {
                "instruction": instruction,
                "input": "Actions: " + ", ".join(actions),
                "output": f"({goal[0]},{goal[1]})"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Dataset saved to {output_path}")
