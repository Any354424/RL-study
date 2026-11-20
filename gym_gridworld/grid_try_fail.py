import numpy as np

from gridworld_env import GridWorldEnv
from gridworld_random_env import GridWorldRandomEnv

env = GridWorldRandomEnv()
obs, info = env.reset(seed=0)
#值迭代
side=env.side
v=np.zeros((side*side,))
v_old=np.full((side*side,), -np.inf)

while True:
    for i in range(side*side):
        action_value=[]
        for a in range(0,5):
            obs, reward, terminated, truncated, info = env.step(a)
            action_value.append(reward)
        best_action_value=max(action_value)
        v[i]=best_action_value
        env.reset()
    count = 0
    for j in range(side*side):
        if abs(v[j]-v_old[j])<1e-6:
            count+=1
    if count == side*side:
        break
    v_old=v.copy()

for n in range(side):
    for m in range(side):
        print(f' {v[n*side+m]} ',end='')
    print('\n')

'''
对每个状态s、每个动作a，调env.step(a)拿reward，取最大值
这是之前的思路
但是这里边使用step传入action会导致state发生变化，没法对单个state多次求动作取最大值
而且没法指定到某个state
环境封装之后，他好像只提供交互结果，自由度/透明度相比之前的下降了很多
也许可以让他采取反方向
'''