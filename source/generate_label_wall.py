import os
import json
import random
from maze_wall import SimpleEnv
from train_simple import simple_greedy_policy

# === 可調參數 ===
SIZE = 10
WALL_RATIO = 0.25
OUTPUT_DIR = "data"
OUTPUT_FILE = f"test_wall_{SIZE}x{SIZE}_SG.jsonl"
NUM_SAMPLES = 200

START_POSITIONS = [(0, 0), (9, 9), (0, 9), (9, 0), (5, 5)]

instruction_pool = [
    "You are in a 10x10 grid. 0 = free, 1 = wall, S = start, G = final position. Follow the actions and mark the final position."
]


def render_wall_map_with_marks(size, walls, start, final):
    grid = [["0" for _ in range(size)] for _ in range(size)]
    for (x, y) in walls:
        if 0 <= x < size and 0 <= y < size:
            grid[x][y] = "1"
    sx, sy = start
    fx, fy = final
    grid[sx][sy] = "S"
    grid[fx][fy] = "G"
    return "\n".join("".join(row) for row in grid)


def generate_cot_trace(start, actions, grid_size, walls):
    x, y = start
    steps = [f"Start at {start}"]

    for i, (dx, dy) in enumerate(actions):
        nx, ny = x + dx, y + dy
        move_info = f"Step {i+1}: move ({dx},{dy}) → "

        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            move_info += f"({x},{y}) ❌ out of bounds"
            steps.append(move_info)
            break

        elif (nx, ny) in walls:
            move_info += f"({x},{y}) ❌ wall"
            steps.append(move_info)
            break

        else:
            x, y = nx, ny
            move_info += f"({x},{y}) ✅ ok"
            steps.append(move_info)

    steps.append(f"Final position: ({x},{y})")
    return "\n".join(steps), (x, y)


def generate_with_walls(n_samples):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    for i in range(n_samples):
        seed = 6000 + i
        env = SimpleEnv(size=SIZE, seed=seed, wall_ratio=WALL_RATIO)
        start = random.choice(START_POSITIONS)
        goal = env.get_goal()
        full_actions = simple_greedy_policy(start, goal)
        actions = [eval(a["vec"]) for a in full_actions]

        if i % 3 == 0 and len(actions) > 1:
            actions = actions[:-1]

        cot_trace, final_pos = generate_cot_trace(
            start, actions, SIZE, env.walls)
        wall_map = render_wall_map_with_marks(
            SIZE, env.walls, start, final_pos)

        input_text = (
            f"Map:\n{wall_map}\n"
            f"Actions: " + ", ".join([str(a) for a in actions])
        )

        output_text = cot_trace + "\n\nMap with final position:\n" + wall_map

        record = {
            "instruction": random.choice(instruction_pool),
            "input": input_text,
            "output": output_text
        }

        all_data.append(record)

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 完成：{n_samples} 筆資料儲存至 {save_path}")


if __name__ == "__main__":
    generate_with_walls(NUM_SAMPLES)
