import numpy as np
import random
from gridworld_random_env import GridWorldRandomEnv
import math

side= 10  #修改模型
random_start  = True  #是否随机起点
boundary_punish=-1  #边界惩罚
gamma=0.9      #衰减因子
eps=0.3         #初始eps
episode_num=500    #采样次数
max_steps=250    #最大步数
random.seed(0)      #种子
env = GridWorldRandomEnv(   #创建模型
    side=side,
    boundary_punish=boundary_punish,
    traps={4: -1},
    max_steps=max_steps,
    random_start=random_start)
env.reset(seed=0)
start_policy = {s: random.randrange(5) for s in range(side*side-1)}    #选定一个初始策略

def eps_action(action,eps):     #ai的神秘简化版
        if random.random() < eps:
            return random.randrange(5)  # 以ε概率均匀随机,这边假设是0.2
        return action  # 以1-ε概率走 greedy，也就是0.8，加上上面的0.2/5=0.04刚好是0.84

#MC basic，每个s，a都取一遍
def mc_sample(episode_num,policy,mode):
    action_value=np.zeros((side*side-1,5))      #准备最后的行动价值列表
    sum_value=np.zeros((side*side-1,5))     #准备每一个sa的总和
    run_count=np.zeros((side*side-1,5))     #对应sa的遍历数
    for episode in range(episode_num):     #定义采样次数
        start_idx,_=env.reset()     #每轮开始更新
        start_action=eps_action(policy[start_idx],eps)      #选择eps动作
        obs, reward, terminated, truncated, info = env.step(start_action)  # 走第一步
        trajectory = []  # 记录总路径
        trajectory.append((start_idx,start_action, reward))
        while not (terminated or truncated):    #没有结束或者到达目的地
            obs_past=obs    #记录当前位置
            a = eps_action(policy[obs_past], eps)
            obs, reward, terminated, truncated, info = env.step(a)   #走
            trajectory.append((obs_past,a, reward))  #添加记录
        if mode=='basic':
            times = 0
            G = 0
            for s, a, reward in trajectory:
                G += reward * gamma ** times
                times += 1      #记录数量
            sum_value[start_idx][start_action] += G
            run_count[start_idx][start_action] += 1
        if mode == 'every':
            G = 0
            for s, a, reward in reversed(trajectory):
                G = G* gamma+reward
                sum_value[s][a] += G
                run_count[s][a] += 1
        if mode == 'first':
            visit=set()
            G = 0
            for s, a, reward in reversed(trajectory):
                G = G * gamma + reward
                if (s,a) in visit:
                    continue
                else:
                    sum_value[s][a] += G
                    run_count[s][a] += 1
                visit.add((s,a))
    for s in range(side * side - 1):
        for a in range(5):
            if run_count[s][a] > 0:
                action_value[s][a] = sum_value[s][a] / run_count[s][a]
            else:
                 # 从未访问过，根据你的算法给一个默认值
                action_value[s][a] = -math.inf  #
    return action_value,run_count
#策略更新
policy=start_policy.copy()
for i in range(10):
    Q,run_count = mc_sample(episode_num,policy,'first')
    new_policy = {s: int(np.argmax(Q[s])) for s in range(side*side-1)}
    eps = max(0.05, eps * 0.8)
    if new_policy == policy:
        break
    policy = new_policy

arrows = ["↑", "→", "↓", "←", "·"]      #涉及直观的策略
for i in range(side):
    for j in range(side):
        idx = i * side + j
        if idx == side * side - 1:      # 终点
            print(" G ", end="")
        else:
            print(f" {run_count[idx].sum():3.0f}{arrows[policy[idx]]} ", end="")
    print()

#数据统计
Q_again, rc_eval = mc_sample(episode_num, policy, 'first')   #再跑一遍，测试两次的差别

n_states = side * side - 1  #states数量
min_count = 5      # 最低采样次数
margin = 0.1       # 最优 Q 比次优 Q 至少高 0.1
covered = stable = confident = 0
for s in range(n_states):
    a = policy[s]
    # 这个状态完全没采到，直接跳
    if np.all(np.isneginf(Q_again[s])):  #如果这个state的所有action value都是负无穷（初始值），判断为没遇到
        continue
    # 箭头动作样本太少，不可信，跳
    if rc_eval[s][a] < min_count:
        continue
    covered += 1
    #如果第二次运行得到的最大action value对应的action不是策略里面的，说明不稳定
    if int(np.argmax(Q_again[s])) != a:
        continue
    stable += 1
    #领先第二名足够多（只比较有限值，排除 -inf）
    finite = Q_again[s][np.isfinite(Q_again[s])]    #isfinite：生成掩码：正常数为 True，±inf、nan为 False
    if len(finite) >= 2:    #剩下的还有两个以上
        top2 = np.sort(finite)[-2:]     #取出最大的两个
        gap = top2[1] - top2[0]     #相减
    else:
        gap = -np.inf   #默认为最小的
    if gap >= margin:
        confident += 1  #差距足够大

print(f"covered  : {covered}/{n_states}")
print(f"stable   : {stable}/{n_states}")
print(f"confident: {confident}/{n_states}")
print(f"unvisited: {n_states - covered}/{n_states}")

def success_rate(env, policy, n_states, max_steps=200):     #用确定greedy策略
    ok = 0
    for s in range(n_states):
        obs, _ = env.reset(options={"start_state": s})
        for _ in range(max_steps):
            obs, _, term, trunc, _ = env.step(policy[obs])
            if term or trunc:
                break
        ok += int(term)
    return ok / n_states

print(f"success_rate={success_rate(env, policy, n_states):.1%}")





