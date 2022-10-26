"""This file implements a wrapper that adds an action, which controls the speed of the agent.

Author: Jakob Thumm
Date: 26.10.2022
"""

import gym
import gym.spaces as spaces
import numpy as np


class SpeedActionWrapper(gym.ActionWrapper):
    """This class implements a wrapper that adds the speed action to the action space."""

    def __init__(self, env):
        """Initialize the wrapper."""
        super().__init__(env)
        action_space = env.action_space
        assert isinstance(action_space, spaces.Box), "The action space must be continuous."
        low = np.append(action_space.low, -1)
        high = np.append(action_space.high, 1)
        self.action_space = spaces.Box(low, high)

    def action(self, act):
        """Return the action scaled by the speed action."""
        a = act[:-1]
        speed = (act[-1] + 1) / 2
        return a * speed

    def step(self, action):
        """Wrap the step function with the scaled action.

        Checks if there is another replaced aciton in the info dict.
        """
        a = self.action(action)
        obs, reward, done, info = self.env.step(a)
        if "action" in info:
            info["action"] = np.append(info["action"], 1)
        return obs, reward, done, info
