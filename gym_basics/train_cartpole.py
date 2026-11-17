"""
CartPole + PPO 完整示例
运行: python gym_basics/train_cartpole.py
"""

import os
import numpy as np      #导入numpy
import gymnasium as gym     #导入gym
import matplotlib.pyplot as plt     #导入绘图工具
from stable_baselines3 import PPO   #导入算法PPO
from stable_baselines3.common.callbacks import BaseCallback     #导入一个类
from stable_baselines3.common.evaluation import evaluate_policy #导入评估函数
from stable_baselines3.common.monitor import Monitor    #导入监视器

# ---------- 配置 ----------
TOTAL_STEPS = 50_000    #agent和环境交互50000次
SEED = 0        #选择一个种子
OUT_DIR = "results"     #设置输出文件夹目录叫results
MODEL_PATH = os.path.join(OUT_DIR, "ppo_cartpole")   #在results里保存为ppo_cartpole
FIG_PATH = os.path.join(OUT_DIR, "cartpole_curve.png")  #在results里保存为cartpole_curve.png


class RewardRecorder(BaseCallback):     #定义一个记录类
    """每轮 rollout 结束，记录最近 100 局的平均回报。"""

    def __init__(self):
        super().__init__()  #传入BaseCallback的属性
        self.steps = []     #记录步数(图的横坐标)
        self.rewards = []   #记录平均分列表（图的纵坐标）

    def _on_step(self) -> bool: #用于判断是否继续走
        return True

    def _on_rollout_end(self) -> None:  #收集完一批经验之后调用的方法
        buf = self.model.ep_info_buffer #从模型中取出统计记录
        if len(buf) > 0:    #至少进行了一轮
            self.steps.append(self.num_timesteps)   #添加步数
            self.rewards.append(float(np.mean([ep["r"] for ep in buf])))
            #对buf里面遍历元素reward，求出平均值，转为浮点数加到rewards列表里面


def main():
    os.makedirs(OUT_DIR, exist_ok=True)     #创建out_dir，也就是result目录

    # 1. 环境
    env = Monitor(gym.make("CartPole-v1"))  #创建CartPole-v1环境，外面套上监视器

    # 2. 模型
    model = PPO(        #创建模型，调用PPO的构造函数
        "MlpPolicy",
        env,        #传入环境
        seed=SEED,      #传入种子
        verbose=1,      #不用管
        learning_rate=3e-4,     #传入学习率
        n_steps=2048,       #跑出2048条记录
        batch_size=64,      #每次抓64个
        gamma=0.99,     #传入折扣因子
        tensorboard_log="./tb/",        #不用管
    )#这里创建了一个神经网络模型，之后的任务就是根据环境给的反馈修改里面的参数

    # 3. 训练
    recorder = RewardRecorder()  #创建记录器实例
    model.learn(total_timesteps=TOTAL_STEPS, callback=recorder) #收集totalsteps步，这里一次跑2048步，最后一步会收集完，所以超了一点
    #传入callback参数为之前写的
    model.save(MODEL_PATH)  #通过指定的路径保存

    print(f"\n记录点数: {len(recorder.steps)}")  #50000/2048=24.4所以是25不是24
    #记录total——steps后走了几步
    # 4. 评估
    eval_env = Monitor(gym.make("CartPole-v1"))  #再创建一个环节
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    #从sb3调用的，给模型，环境，运行轮数，返回平均奖励和标准差。这里是跑了20局，就是正常500结束，或者越界结束
    print(f"评估回报: {mean_reward:.1f} ± {std_reward:.1f}")

    # 5. 画图
    plt.figure(figsize=(8, 4))
    plt.plot(recorder.steps, recorder.rewards, linewidth=1.5, marker="o", markersize=3)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean episode reward (last 100 episodes)")
    plt.title("PPO on CartPole-v1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    print(f"曲线已保存: {FIG_PATH}")


if __name__ == "__main__":
    main()