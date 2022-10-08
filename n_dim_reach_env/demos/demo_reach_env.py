"""Simple demo function of Reach Env."""
import gym
from n_dim_reach_env.envs.reach_env import ReachEnv  # noqa: F401
from gym.wrappers import TimeLimit
env = gym.make("n_dim_reach_env/ReachEnv-v0",
               n_dim=2,
               max_action=0.05,
               goal_distance=0.01,
               done_on_collision=False,
               randomize_env=True,
               collision_reward=-1.0,
               goal_reward=1.0,
               step_reward=-1.0,
               reward_shaping=True,
               render_mode=None,
               seed=42)
env = TimeLimit(env, 1000)
obs, info = env.reset()

for i in range(int(1e6)):
    a = env.action_space.sample()
    o, r, d, info = env.step(a)
    if d:
        o, info = env.reset()
print("done")
