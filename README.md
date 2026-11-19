# 这是我的强化学习仓库

由于目前系统性学习的主要是西湖大学的强化学习课程，这里主要实践的是它里面的例子网格世界
目前的学习进度：
	西湖大学《强化学习的数学原理》第 1–4 章
		基础概念
		贝尔曼公式
		贝尔曼最优公式
		值迭代与策略迭代
	Python代码
		Gymnasium使用
		PPO初步应用
	Mujoco
		下载了`MS-Human-700`
	![[a2a3fac7-0f2b-4b9d-956e-aaeae8e9c2f0.png]]
之后打算看看代码的资料，比如强化学习的系统性代码练习
 
---

## 代码顺序

| 阶段  | 文件                                      | 内容                                                                                                |
| --- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 1   | `gridworld/return_random.py`            | 随机走一条路，回报是多少，主要是把网课抽象的东西落到代码上面                                                                    |
| 2   | `gridworld/The_world.py`                | 为了方便之后的代码，把世界设定单独抽出来作为模块                                                                          |
| 3   | `gridworld/just_try.py`                 | 纯暴力乱找多次尝试，多次运行结果相同，认为找到了最优路径                                                                      |
| 4   | `gridworld/value_iteration.py`          | 写了一个值迭代的代码，和之前的结论在数字上相同，验证了值迭代的有效性                                                                |
| 5   | `gridworld/policy_iteration.py`         | 策略迭代，返回总次数和策略迭代次数                                                                                 |
| 6   | `gym_gridworld/gridworld_env.py`        | 初步接触gym，尝试让之前写的网格世界根据[Env - Gymnasium 文档 - Gymnasium 文档](https://gymnasium.org.cn/api/env/)变成标准环境 |
| 7   | `gym_gridworld/grid_try.py`             | 调用自己写的环境，让他走了指定的一条路，反馈和调用没有问题                                                                     |
| 8   | `gym_gridworld/grid_try_fail.py`        | 试图用环境写一个值迭代，但是由于环境似乎不支持值迭代所需的动作和反馈，最后放弃去看PPO了，这是环境的标准用法                                           |
| 9   | `gym_basics/train_cartpole.py`          | 让AI写了一个先拿别人的环境跑通的流程                                                                               |
| 10  | `gym_gridworld/gridworld_random_env.py` | 根据需求扩写了之前的那个环境，并添加了许多可以修改的值，使得调用的时候有更多可选                                                          |
| 11  | `gym_gridworld/PPO_gridworld.py`        | 让 PPO 自己学习最优路径，可视化了结果                                                                             |


---
