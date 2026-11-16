from gridworld_env import GridWorldEnv

env = GridWorldEnv()
obs, info = env.reset(seed=0)
print("起点:", obs)

# 手写一条路：右、右、下、下  → s1→s2→s3→s6→s9
path = [2,2,1,1]
total = 0.0
gamma = 0.9

for t, a in enumerate(path):
    obs, r, terminated, truncated, info = env.step(a)
    total += (gamma ** t) * r
    print(f"t={t} 动作={a} 到={obs} 奖励={r:+.1f} 结束={terminated}")
    env.render()

print(f"折扣回报: {total:.3f}")
