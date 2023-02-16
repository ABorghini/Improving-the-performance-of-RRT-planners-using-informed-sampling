from env import *

class EnvKino(Env):
    def __init__(self, x_start, x_goal, w = 100, h = 100, thickness = 1, delta = 0.5):
        super().__init__(x_start, x_goal, w, h, thickness, delta)


    def is_inside_obs(self, state):
        delta = 0.5

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= state[0] - (x - delta) <= w + 2 * delta \
                    and 0 <= state[1] - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= state[0] - (x - delta) <= w + 2 * delta \
                    and 0 <= state[1] - (y - delta) <= h + 2 * delta:
                return True

        return False


class NodeKino():
    def __init__(self, n):
        self.node = Matrix(n)
        self.cost = np.inf
        self.time = np.inf
        self.cost2goal = np.inf
        self.time2goal = np.inf
        self.near_goal = False
        self.parent = None
        self.children = []
