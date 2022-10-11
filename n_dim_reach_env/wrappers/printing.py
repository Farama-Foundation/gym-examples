"""Wrapper for prinitng the time steps of an environment."""
from typing import Optional

import gym


class PrintingWrapper(gym.Wrapper):
    """This wrapper will print every nth step."""

    def __init__(
        self,
        env: gym.Env,
        print_every: int = 100,
    ):
        """Initializes the :class:`PrinintWrapper` wrapper with an environment.
        Args:
            env: The environment to apply the wrapper
            print_every: Print every n-th step
        """
        super().__init__(env)
        self._print_every = print_every
        self.printed = False

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.
        Args:
            action: The environment step action
        Returns:
            The environment step ``(observation, reward, done, info)``
        """
        observation, reward, done, info = self.env.step(action)

        if not self.printed:
            print("New episode")
            self.printed = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.
        Args:
            **kwargs: The kwargs to reset the environment with
        Returns:
            The reset environment
        """
        self.printed = False
        return self.env.reset(**kwargs)
