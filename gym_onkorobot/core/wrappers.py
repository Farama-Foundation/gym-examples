# import gymnasium as gym
# import math
#
#
# class ActionBonus(gym.Wrapper):
#
#     def __init__(self, env):
#         """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
#
#         Args:
#             env: The environment to apply the wrapper
#         """
#         super().__init__(env)
#         self.counts = {}
#
#     def step(self, action):
#         """Steps through the environment with `action`."""
#         obs, reward, terminated, truncated, info = self.env.step(action)
#
#         env = self.unwrapped
#         tup = (tuple(env.agent_pos), env.agent_dir, action)
#
#         # Get the count for this (s,a) pair
#         pre_count = 0
#         if tup in self.counts:
#             pre_count = self.counts[tup]
#
#         # Update the count for this (s,a) pair
#         new_count = pre_count + 1
#         self.counts[tup] = new_count
#
#         bonus = 1 / math.sqrt(new_count)
#         reward += bonus
#
#         return obs, reward, terminated, truncated, info
#
#
# class PositionBonus(gym.Wrapper):
#     """
#     Adds an exploration bonus based on which positions
#     are visited on the grid.
#     """
#
#     def __init__(self, env):
#         """A wrapper that adds an exploration bonus to less visited positions.
#
#         Args:
#             env: The environment to apply the wrapper
#         """
#         super().__init__(env)
#         self.counts = {}
#
#     def step(self, action):
#         """Steps through the environment with `action`."""
#         obs, reward, terminated, truncated, info = self.env.step(action)
#
#         # Tuple based on which we index the counts
#         # We use the position after an update
#         env = self.unwrapped
#         tup = tuple(env.agent_pos)
#
#         # Get the count for this key
#         pre_count = 0
#         if tup in self.counts:
#             pre_count = self.counts[tup]
#
#         # Update the count for this key
#         new_count = pre_count + 1
#         self.counts[tup] = new_count
#
#         bonus = 1 / math.sqrt(new_count)
#         reward += bonus
#
#         return obs, reward, terminated, truncated, info