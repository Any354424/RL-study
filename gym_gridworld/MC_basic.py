import numpy as np
import random
from gridworld_random_env import GridWorldRandomEnv

side= 3  #修改模型
random_start  = True  #是否随机起点
boundary_punish=-1  #边界惩罚
max_steps=50    #最大步数
gamma=0.9
env = GridWorldRandomEnv(   #创建模型
    side=side,
    boundary_punish=boundary_punish,
    traps={4: -1},
    max_steps=max_steps,
    random_start=random_start)
start_policy={s:0 for s in range(side*side-1)}    #选定一个初始策略

#MC basic，每个s，a都取一遍
def mc_sample(episode_num,policy):
    action_value=np.zeros((side*side-1,5))      #准备最后的行动价值列表
    sum_value=np.zeros((side*side-1,5))     #准备每一个sa的总和
    run_count=np.zeros((side*side-1,5))     #对应sa的遍历数
    for episode in range(episode_num):     #定义采样次数
        #以下是exploring start
        s0=random.randint(0,side*side-2)    #随机s
        a0=random.randint(0,4)      #随机a
        env.reset(options={'start_state':s0})   #设置环境
        obs,reward,terminated,truncated,info=env.step(a0)   #走第一步
        trajectory=[]   #记录总路径
        trajectory.append((s0,a0,reward))       #写第一步
        while not (terminated or truncated):    #没有结束或者到达目的地
            obs_past=obs    #记录当前位置
            obs, reward, terminated, truncated, info = env.step(policy[obs_past])   #走
            trajectory.append((obs_past,policy[obs_past], reward))  #添加记录
        times = 0
        G = 0
        for s, a, reward in trajectory:
            G += reward * gamma ** times
            times += 1      #记录数量
        sum_value[s0][a0] += G
        run_count[s0][a0] += 1
    for s in range(side*side-1):
        for a in range(5):
            action_value[s][a]=sum_value[s][a]/run_count[s][a]
    return action_value
#策略更新
policy=start_policy.copy()
while True:
    Q = mc_sample(1000,policy)
    new_policy = {s: int(np.argmax(Q[s])) for s in range(side*side-1)}
    if new_policy == policy:
        break
    policy = new_policy
print(policy)










