import argparse
import os
import json
from maze_simple import SimpleEnv
from train_simple import simple_greedy_policy

START_POSITIONS = [
    (4, 4), (4, 5), (4, 6),
    (5, 4), (5, 5), (5, 6),
    (6, 4), (6, 5), (6, 6),
    (0, 0), (10, 10)
]


def render_map_with_start(size, start):
    grid = [["0" for _ in range(size)] for _ in range(size)]
    x, y = start
    grid[x][y] = "S"
    return "\n".join("".join(row) for row in grid)


def render_map_with_goal(size, start, goal):
    grid = [["0" for _ in range(size)] for _ in range(size)]
    sx, sy = start
    gx, gy = goal
    grid[sx][sy] = "S"
    grid[gx][gy] = "G"
    return "\n".join("".join(row) for row in grid)


def generate_cot_steps(start, actions):
    x, y = start
    steps = [f"Start at ({x},{y})"]
    for i, (dx, dy) in enumerate(actions):
        x += dx
        y += dy
        steps.append(f"Step {i+1}: ({x-dx},{y-dy}) + ({dx},{dy}) = ({x},{y})")
    steps.append(f"Final position: ({x},{y})")
    return "\n".join(steps), (x, y)


def generate_cot_dataset(seed_start, seed_end, size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"cot_map_{size}x{size}_{seed_start}to{seed_end}.jsonl"
    output_path = os.path.join(output_dir, output_name)

    instruction_text = "Given a map, where 0 is path and S is the starting point, follow the (dx, dy) actions step-by-step to determine the final position."

    with open(output_path, "w", encoding="utf-8") as f:
        for start in START_POSITIONS:
            for seed in range(seed_start, seed_end + 1):
                env = SimpleEnv(size=size, seed=seed)
                goal = env.get_goal()
                path = simple_greedy_policy(start, goal)
                actions = [eval(p["vec"]) for p in path]

                cot_trace, final_pos = generate_cot_steps(start, actions)
                map_input = render_map_with_start(size, start)
                map_output = render_map_with_goal(size, start, final_pos)

                record = {
                    "instruction": instruction_text,
                    "input": f"Start at ({start[0]},{start[1]})\nMap:\n{map_input}\nActions: {', '.join(str(a) for a in actions)}",
                    "output": cot_trace + "\n\nMap with SG:\n" + map_output
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
