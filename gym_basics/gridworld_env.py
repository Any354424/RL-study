import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """3x3 网格：s7 是陷阱，s9 是终点。
    s1 s2 s3
    s4 s5 s6
    s7 s8 s9
    """
    # metadata = {"render_modes": ["human"], "render_fps": 4}

    # 格子说明（按编号 0-8）
    CELL_REWARD = [0, 0, 0, 0, 0, 0, -1, 0, 1]
    START = 0
    GOAL = 8

    # 动作：0上 1右 2下 3左 4不动
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    def __init__(self, max_steps=50, render_mode=None):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(9)
        self.action_space = spaces.Discrete(5)
        self._pos = self.START
        self._steps = 0

    # ---------- 工具 ----------
    def _rc(self, idx):
        return divmod(idx, 3)          # 给出编号，返回(行, 列)

    def _idx(self, r, c):
        return r * 3 + c               # 给行列坐标，返回编号

    def _obs(self):
        return int(self._pos)

    # ---------- 核心接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.START
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        r, c = self._rc(self._pos)
        dr, dc = self.MOVES[action]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < 3 and 0 <= nc < 3):   # 撞墙
            nr, nc = r, c
            reward = -1.0
        else:
            reward = float(self.CELL_REWARD[self._idx(nr, nc)])

        self._pos = self._idx(nr, nc)
        self._steps += 1

        terminated = self._pos == self.GOAL          # 到终点
        truncated = self._steps >= self.max_steps    # 超时

        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        for r in range(3):
            row = []
            for c in range(3):
                i = self._idx(r, c)
                row.append("[A]" if i == self._pos else f" s{i+1} ")
            print("".join(row))