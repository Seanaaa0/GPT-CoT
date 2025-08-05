import numpy as np


class SimpleEnv:
    def __init__(self, size=5, seed=None, wall_ratio=0.1):
        self.size = size
        self.start = (0, 0)
        self.rng = np.random.RandomState(seed)
        self.goal = self._generate_goal()
        self.wall_ratio = wall_ratio
        self.walls = self._generate_walls()
        self.agent_pos = list(self.start)
        self.done = False

    def _generate_goal(self):
        while True:
            goal = (self.rng.randint(0, self.size),
                    self.rng.randint(0, self.size))
            if goal != self.start:
                return goal

    def _generate_walls(self):
        num_walls = int(self.size * self.size * self.wall_ratio)
        walls = set()
        attempts = 0
        while len(walls) < num_walls and attempts < 1000:
            i = self.rng.randint(0, self.size)
            j = self.rng.randint(0, self.size)
            if (i, j) not in [self.start, self.goal] and (i, j) not in walls:
                walls.add((i, j))
            attempts += 1
        return walls

    def reset(self):
        self.agent_pos = list(self.start)
        self.done = False
        return tuple(self.agent_pos)

    def step(self, action):
        if self.done:
            return tuple(self.agent_pos), True

        x, y = self.agent_pos
        new_x, new_y = x, y

        if action == "up" and x > 0:
            new_x -= 1
        elif action == "down" and x < self.size - 1:
            new_x += 1
        elif action == "left" and y > 0:
            new_y -= 1
        elif action == "right" and y < self.size - 1:
            new_y += 1

        if (new_x, new_y) not in self.walls:
            self.agent_pos = [new_x, new_y]

        self.done = tuple(self.agent_pos) == self.goal
        return tuple(self.agent_pos), self.done

    def get_goal(self):
        return self.goal

    def get_walls(self):
        return self.walls
