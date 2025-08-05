import os
import json
import random
from pathlib import Path
from collections import deque

SIZE = 11
WALL_RATIO = 0.25
NUM_SAMPLES = 100
OUTPUT_PATH = "data/vec_wall_11x11_SG.jsonl"

START_POSITIONS = [(0, 0), (10, 10), (0, 10), (10, 0), (5, 5)]


def render_wall_map_input(size, walls, start):
    grid = [["0" for _ in range(size)] for _ in range(size)]
    for (x, y) in walls:
        if 0 <= x < size and 0 <= y < size:
            grid[x][y] = "1"
    sx, sy = start
    grid[sx][sy] = "S"
    return "\n".join("".join(row) for row in grid)


def render_wall_map_with_goal(size, walls, start, final):
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
        move_info = f"Step {i+1}: move ({dx},{dy}) → ({nx},{ny}) ✅ ok"
        steps.append(move_info)
        x, y = nx, ny
    steps.append(f"Final position: ({x},{y})")
    return "\n".join(steps), (x, y)


def generate_random_wall_env(size, wall_ratio, seed):
    random.seed(seed)
    wall_count = int(size * size * wall_ratio)
    all_positions = [(x, y) for x in range(size) for y in range(size)]
    walls = set(random.sample(all_positions, wall_count))
    while True:
        goal = random.choice(all_positions)
        if goal not in walls:
            return walls, goal


def bfs(start, goal, walls, size):
    visited = set()
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in walls and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
    return []


def generate_samples():
    instruction_text = f"You are in a {SIZE}x{SIZE} grid. 0 = free, 1 = wall, S = start, G = final position. Follow the actions and mark the final position."
    records = []
    for i in range(NUM_SAMPLES):
        seed = 8000 + i
        start = random.choice(START_POSITIONS)
        walls, goal = generate_random_wall_env(SIZE, WALL_RATIO, seed)
        path = bfs(start, goal, walls, SIZE)
        if not path:
            continue
        actions = path  # 保留完整合法路徑

        cot_trace, final_pos = generate_cot_trace(start, actions, SIZE, walls)
        wall_map_input = render_wall_map_input(SIZE, walls, start)
        wall_map_output = render_wall_map_with_goal(
            SIZE, walls, start, final_pos)

        record = {
            "instruction": instruction_text,
            "input": f"Map:\n{wall_map_input}\nActions: " + ", ".join([str(a) for a in actions]),
            "output": cot_trace + "\n\nMap with final position:\n" + wall_map_output
        }
        records.append(record)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(records)} samples at: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_samples()
