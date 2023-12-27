import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=123, options={})

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()