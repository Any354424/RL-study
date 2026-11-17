import gymnasium as gym

env = gym.make("HalfCheetah-v5", render_mode="human")
obs, _ = env.reset(seed=0)
for _ in range(2000):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()