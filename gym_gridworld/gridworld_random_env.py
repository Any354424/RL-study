import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldRandomEnv(gym.Env):
    """3x3 网格：s7 是陷阱，s9 是终点。
    s1 s2 s3
    s4 s5 s6
    s7 s8 s9
    """
    # metadata = {"render_modes": ["human"], "render_fps": 4}

    # ---------- 工具 ----------
    def _rc(self, idx):
        return divmod(idx, self.side)  # 给出编号，返回(行, 列)

    def _idx(self, r, c):
        return r * self.side + c  # 给行列坐标，返回编号

    def _obs(self):  # 返回观测值（位置），这里单独列出来而不是使用._pos是为了方便未来对观察进行操作，说不定不要位置了
        return int(self._pos)  # 加一个int防止意外

    #下面开始定义世界

    #格子说明（按编号 0-side*side）起始位置，目标规划，陷阱惩罚,下标从0开始
    start_idx= 0
    # 动作：0上 1右 2下 3左 4不动
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    #传入提醒：traps为一个{位置：惩罚}的字典，起点是一个数字下标
    def __init__(self,side=3,max_steps=50,  #环境基础设置
                 traps=None,trap_punish=-1,boundary_punish=-1,  #陷阱与边界惩罚
                 random_start=False,random_state_transfer=False,    #随机开关：起点，状态转移，奖励
                 render_mode=None,):
        super().__init__()
        self.max_steps = max_steps  #规定最大迭代次数
        self.render_mode = render_mode  #渲染？可视化的什么模式
        self.random_start=random_start  #是否开始随机起点
        self.side=side  #网格边长
        self.n = side * side    #网格数
        self.goal_state = self.n - 1    #目标在右下角
        self.boundary_punish=boundary_punish
        self.trap_punish=trap_punish     #陷阱惩罚

        #下面是新加的可选陷阱，陷阱以{下标：惩罚}的字典传入
        if traps is None:
            traps = {side * (side - 1): trap_punish}
        self.traps = dict(traps)
        if self.goal_state in self.traps:
            raise ValueError("陷阱不能放在终点格")
        for idx in self.traps:
            if not 0 <= idx < self.n:
                raise ValueError(f"陷阱下标越界：{idx}")

        self.observation_space = spaces.Discrete(self.side * self.side)     #可观测到的集合，一个9
        self.action_space = spaces.Discrete(5)  #行动的集合，一共5
        self.cell_reward = np.zeros((side*side,),dtype=float)   #把格子的奖励移到这里面，如果需要修改某一个实例的奖励，避免污染
        self.cell_reward[self.goal_state]=1     #添加目的地奖励
        for idx, r in self.traps.items():   #添加陷阱格惩罚
            self.cell_reward[idx]=r
        self._pos = self.start_idx  #初始观察到的位置为start_idx
        self._steps = 0  #给初始步数0
        #上面那两个其实可以随便写，为了防止在reset前就被调用，这里选这两个参数是相当于提前reset了一次

    # ---------- 核心接口 ----------
    def reset(self, seed=None, options=None): #重置步数step,返回原位
        super().reset(seed=seed)
        options = options or {}
        if "start_state" in options:    # MC Basic 指定了起点
            self._pos = int(options["start_state"])     #输入{start_state：0}这样的字典
        elif self.random_start:  # 随机起点
            self._pos = int(self.np_random.integers(0, self.n - 1))
        else:
            self._pos = self.start_idx  #将位置转为初始位置start_index
        self._steps = 0     #重置步数step
        return self._obs(), {}  #返回初始化的位置

    def step(self, action):     #传入行动
        r, c = self._rc(self._pos)      # 给出位置编号，返回(行, 列)
        dr, dc = self.moves[action]     #选择行动对应的元组
        nr, nc = r + dr, c + dc         #元组相加，得到理论上的下一个位置

        if not (0 <= nr < self.side and 0 <= nc < self.side):   # 撞墙
            nr, nc = r, c       #撤销运动
            reward = self.boundary_punish       #给惩罚
        else:
            reward = float(self.cell_reward[self._idx(nr, nc)])    #计算对应行动到达格子给的奖励

        self._pos = self._idx(nr, nc)   #更新位置
        self._steps += 1    #记录运动步数

        terminated = self._pos == self.goal_state          # 到终点
        truncated = (not terminated) and (self._steps >= self.max_steps)     # 超时，且避免以为是单纯时间到了导致的结束，如果到目的地了就不触发

        return self._obs(), reward, terminated, truncated, {} #依次返回位置（），奖励，是否到目的地，是否超时

    def render(self):   #可视化目前的位置
        for r in range(self.side):  #先axit=0方向
            row = []
            for c in range(self.side):  #计算每一行
                i = self._idx(r, c)     #已经固定了行的坐标，遍历列数来遍历元素
                row.append("[A] " if i == self._pos else f" s{i+1} ")  #如果到达目前位置，输出A，不然旧原样返回
            print("".join(row))  #打印side次，输出完整图表