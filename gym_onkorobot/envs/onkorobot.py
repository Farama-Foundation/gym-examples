from typing import Optional, Union, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import Box, Discrete
from gym_onkorobot.core.actions import Actions
from gym_onkorobot.utils.window import Window

ACTION_SHAPE = (3, 1)


class OnkoRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 max_steps: int = 100,
                 render_mode=None,
                 point_cloud=None):
        # TODO:
        self.max_steps = max_steps
        self.max_steps = None
        self.step_count = 0
        cloud_space = spaces.Box(
            low=0,
            high=255,
            shape=(100, 100, 100, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict({"cloud": cloud_space})
        # action description
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_range = (0, 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: Optional[Window] = None
        self.clock = None
        # init grid

    def step(self, action: Actions) -> Tuple[ObsType, float, bool, bool, dict]:
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        obs = self.gen_obs()

        if action == self.actions.down:
            pass
        elif action == self.actions.up:
            pass
        elif action == self.actions.left:
            pass
        elif action == self.actions.right:
            pass
        elif action == self.actions.forward:
            pass
        elif action == self.actions.dose:
            pass
        elif action == self.actions.done:
            raise NotImplemented
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.step_count = 0


    # def _get_obs(self):
    #     # TODO: return point cloud
    #     return {"agent": self._agent_location, "target": self._target_location}
    #
    # def _get_info(self):
    #     # return some usefull information
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }
    #
    # def reset(self, seed=None, options=None):
    #     # We need the following line to seed self.np_random
    #     super().reset(seed=seed)
    #
    #     # Choose the agent's location uniformly at random
    #     self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
    #
    #     # We will sample the target's location randomly until it does not coincide with the agent's location
    #     self._target_location = self._agent_location
    #     while np.array_equal(self._target_location, self._agent_location):
    #         self._target_location = self.np_random.integers(
    #             0, self.size, size=2, dtype=int
    #         )
    #
    #     observation = self._get_obs()
    #     info = self._get_info()
    #
    #     if self.render_mode == "human":
    #         self._render_frame()
    #
    #     return observation, info
    #
    # def step(self, action):
    #     # Map the action (element of {0,1,2,3}) to the direction we walk in
    #     direction = self._action_to_direction[action]
    #     # We use `np.clip` to make sure we don't leave the grid
    #     self._agent_location = np.clip(
    #         self._agent_location + direction, 0, self.size - 1
    #     )
    #     # An episode is done iff the agent has reached the target
    #     terminated = np.array_equal(self._agent_location, self._target_location)
    #     reward = 1 if terminated else 0  # Binary sparse rewards
    #     observation = self._get_obs()
    #     info = self._get_info()
    #
    #     if self.render_mode == "human":
    #         self.render()
    #
    #     return observation, reward, terminated, False, info
    #
    # def render(self):
    #     pass
    #
    # def close(self):
    #     if self.window is not None:
    #         self.window.close()
    def close(self):
        if self.window is not None:
            self.window.close()

    def gen_obs(self):
        pass
