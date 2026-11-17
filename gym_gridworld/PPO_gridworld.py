import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from gridworld_random_env import GridWorldRandomEnv
side= 5
env = GridWorldRandomEnv(side=side,random_go = False)
model = PPO("MlpPolicy", env, seed=0, verbose=1)
model.learn(total_timesteps=100_000)

#策略概率表
print("\n策略（每行一个格子，每列一个动作）：")   #所谓策略矩阵
for s in range(side*side):  #读取环境里有几个状态
    obs_t = torch.tensor([s], dtype=torch.long, device=model.device)    #神经网络要用张量访问
    dist = model.policy.get_distribution(obs_t)     #从问神经网络actor：这个状态选什么动作
    p = dist.distribution.probs     #从神经网络那边拿来的东西的.distribution是关于概率的，.probs把他转为概率传出
    probs = p.detach().numpy() [0]  #先截断，再转为数组
    print(f"s{s+1}: {np.round(probs, 2)}") #输出编号对应的位置，以及行动概率

#critic的V(s)
print("\nV(s)：")
for s in range(side*side):  #选择下标
    obs_t = torch.tensor([s], dtype=torch.long, device=model.device)    #转为张量
    value_t = model.policy.predict_values(obs_t)    #问critic这个位置的预测值是多少
    v = value_t.item()    #转为数字
    print(f"s{s+1}: {v:+.3f}")

'''
这里写一下对PPO的理解
PPO里面有两个神经网络
一个叫actor，他训练的是 传入一个观察值，选什么动作是最好的，也就是一个选择动作的概率
一个是critic，他训练的是 传入一个观察值，这个位置动作的奖励（action value）的平均值（state value）
训练过程是这样的，收到一个观测值（这里是所处的位置）actor会经过它的神经网络传出一个动作（这里是上下左右停）
critic会用这个观测值（位置）预测动作的奖励，然后采取actor给的动作，和这个值作比较
如果这个实际的值比critic预测的要大，则actor会获得一个advantage（优势）的值，使概率偏向这个动作，由此训练actor
critic则是训练自己和这个预测值的偏差减小
这里有个疑惑就是拿单个动作和平均期望作训练，怎么能求出平均期望
答案是通过大量的练习，由于MDP的马尔可夫性质，逐渐会体现出一个概率的分布，也就是actor的选项
通过这个分布，宏观上critic就能训练出概率权重*奖励的预测能力，
也就是说，critic实际上预测的是每一个格子的state value，是固定策略下的普通贝尔曼公式的迭代结果，有点像策略迭代的第一步
但是由于actor，也就是策略在不断的改进，最终趋向于贪心策略下的最优Policy
critic最终变成了在最优Policy策略下的最优State value
这又和值迭代，策略迭代的思路不一样了
'''