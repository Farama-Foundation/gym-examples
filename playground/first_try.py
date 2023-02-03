import gym_onkorobot
import gymnasium

env = gymnasium.make("gym_onkorobot/OnkoRobot-v0", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
