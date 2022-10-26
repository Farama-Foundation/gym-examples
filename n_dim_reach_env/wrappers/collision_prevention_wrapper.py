from enum import Enum
from typing import Callable, Optional, Union
import numpy as np
from gym.core import ActionWrapper
from gym import spaces
from gym import Env

from stable_baselines3.common.vec_env import VecEnv

class REPLACEMENT_TYPE(Enum):
    """Define the replacement types.

    0 - Replace - sample
    1 - Replace - failsafe
    2 - Project
    """
    SAMPLE = 0
    FAILSAFE = 1
    PROJECT = 2


class CollisionPreventionWrapper(ActionWrapper):
    """Checks if the given action would result in a collision and replaces the unsafe action with another action."""

    def __init__(self,
                 env: Union[VecEnv, Env],
                 replace_type: int = 0,
                 n_resamples: int = 20,
                 punishment: float = 0.0):
        """Initialize the collision prevention wrapper.

        Args:
            env (gym.env): The gym environment
            replace_type (int):
                0 - Replace - sample
                1 - Replace - failsafe
                2 - Project 
            n_resamples (int): Number of resamples for type 0.
            punishment (float): if (collision prevented): reward += punishment
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"

        super().__init__(env)
        self.collision_check_fn = self.env.unwrapped.collision_check_fn
        self.n_resamples = n_resamples
        if replace_type == REPLACEMENT_TYPE.SAMPLE.value:
            self.replace_action = self.sample
        elif replace_type == REPLACEMENT_TYPE.FAILSAFE.value:
            self.replace_action = self.failsafe
        elif replace_type == REPLACEMENT_TYPE.PROJECT.value:
            self.replace_action = self.project
            self.project_fn = self.env.unwrapped.project_fn
        else:
            raise NotImplementedError
        self.punishment = punishment
        self.action_resamples = 0

    def reset(self):
        """Reset the action resample counter."""
        self.action_resamples = 0
        return self.env.reset()

    def step(self, action):
        """Wrap the step function with the replaced action.

        Adds the new action to the info dict.
        """
        action, replaced = self.action(action)
        obs, reward, done, info = self.env.step(action)
        if replaced:
            reward += self.punishment
            info["action"] = action
        info["action_resamples"] = self.action_resamples
        return obs, reward, done, info

    def action(self, action):
        """Replace the action if a collision is detected."""
        replaced = False
        if self.collision_check_fn(action):
            action = self.replace_action(action)
            self.action_resamples += 1
            replaced = True
        return action, replaced

    def sample(self, action):
        """Replace the action with a random action that is not in collision.

        Try to resample random action for n times, then just use zero action.
        Uses n = self.n_resamples

        Args:
            action (np.array): Action to execute
        Returns:
            action (np.array)
        """
        for _ in range(self.n_resamples):
            action = self.env.action_space.sample()
            if not self.collision_check_fn(action):
                return action
        return self.failsafe(action)

    def failsafe(self, action):
        """Replace the action with a zero action.

        Args:
            action (np.array): Action to execute
        Returns:
            action (np.array)
        """
        return np.zeros([len(action)])

    def project(self, action):
        """Replace the action with the closest safe action.

        Args:
            action (np.array): Action to execute
        Returns:
            action (np.array)
        """
        return self.project_fn(action)
