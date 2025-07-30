import numpy as np


class SimpleEnv:
    def __init__(self, size=5, seed=None):
        self.size = size
        self.start = (0, 0)
        self.rng = np.random.RandomState(seed)
        self.goal = self._generate_goal()
        self.agent_pos = list(self.start)
        self.done = False

    def _generate_goal(self):
        while True:
            goal = (self.rng.randint(0, self.size),
                    self.rng.randint(0, self.size))
            if goal != self.start:
                return goal

    def reset(self):
        self.agent_pos = list(self.start)
        self.done = False
        return tuple(self.agent_pos)

    def step(self, action):
        if self.done:
            return tuple(self.agent_pos), True

        x, y = self.agent_pos
        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < self.size - 1:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < self.size - 1:
            y += 1

        self.agent_pos = [x, y]
        self.done = (x, y) == self.goal
        return tuple(self.agent_pos), self.done

    def get_goal(self):
        return self.goal
