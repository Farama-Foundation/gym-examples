from typing import Optional, Union, List, Tuple

import gymnasium as gym
#from gymnasium import spaces
#import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType, Env
from gymnasium.spaces import Box, Discrete, Dict

from gymnasium.utils.env_checker import check_env

from typing import Any, Callable

from dataclasses import fields

from gymnasium import spaces
from gymnasium.utils import seeding

from gym_onkorobot.core.mission import MissionSpace
from gym_onkorobot.utils.voxel import Voxel
from gym_onkorobot.utils.window import Window
from gym_onkorobot.core.actions import Actions
from gym_onkorobot.core.observation import Observation
from gym_onkorobot.core.configs import GridConfig


class OnkoRobotEnv(Env):
    def __init__(self,
                 max_steps: int = 2000,
                 grid_config: GridConfig = GridConfig(),
                 mission_space: MissionSpace = MissionSpace()):

        self.max_steps = max_steps
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))

        self.mission = ""

        self.obs = Observation(grid_config=grid_config,)

        cloud_space = Box(
            low=0,
            high=255,
            shape=(grid_config.X_SHAPE, grid_config.Y_SHAPE, grid_config.Z_SHAPE, len(fields(Voxel))),
            dtype="int64",
        )

        self.observation_space = Dict(
            {
                "image": cloud_space,
                "mission": mission_space
            }
        )
        self.step_count = 0

    def step(self, action):

        # print(f"SC: [{self.step_count}]")
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        if action == self.actions.downforward:
            self.obs.move([-1, 0, -1])
        elif action == self.actions.downbackward:
            self.obs.move([1, 0, -1])
        elif action == self.actions.downleft:
            self.obs.move([0, -1, -1])
        elif action == self.actions.downright:
            self.obs.move([0, 1, -1])
        elif action == self.actions.upforward:
            self.obs.move([-1, 0, 1])
        elif action == self.actions.upbackward:
            self.obs.move([1, 0, 1])
        elif action == self.actions.upleft:
            self.obs.move([0, -1, 1])
        elif action == self.actions.upright:
            self.obs.move([0, 1, 1])
        elif action == self.actions.left:
            self.obs.move([0, -1, 0])
        elif action == self.actions.right:
            self.obs.move([0, 1, 0])
        elif action == self.actions.forward:
            self.obs.move([-1, 0, 0])
        elif action == self.actions.backward:
            self.obs.move([1, 0, 0])
        elif action == self.actions.dose:
            reward = self.obs.dose()
        # elif action == self.actions.done:
        #    pass
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.obs.grid.is_healed() or self.step_count >= self.max_steps:
            truncated = True
        #    self.render()

        obs = {
            "image": self.obs.get_grid(),
            "mission": self.mission
        }

       # print(f"STEP: {self.step_count}, rew: {reward}")
        return obs, reward, terminated, truncated, {}

    def render(self):
        v, c = self.obs.grid.render_encode()
        w = Window(v, c)
        w.imshow()

    def reset(self,
              *,
              seed: int | None = None,
              options: dict | None = None) -> tuple[ObsType, dict]:
        self.obs.reset()
        self.step_count = 0

        obs = {
            "image": self.obs.get_grid(),
            "mission": self.mission
        }

        return obs, {}

    def _reward(self) -> float:
        return 1 - 0.9 * (self.step_count / self.max_steps)
