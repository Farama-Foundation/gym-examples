from typing import Optional, Union, List, Tuple

import math
import gymnasium as gym
#from gymnasium import spaces
#import pygame
import numpy as np
import pygame
from gymnasium.core import RenderFrame, ActType, ObsType, Env
from gymnasium.spaces import Box, Discrete, Dict

from dataclasses import fields

from gymnasium import spaces

from gym_onkorobot.core.mission import MissionSpace
from gym_onkorobot.utils.voxel import Voxel
from gym_onkorobot.utils.window import Window
from gym_onkorobot.core.actions import Actions
from gym_onkorobot.core.observation import Observation
from gym_onkorobot.core.configs import GridConfig, ObservationConfig


class OnkoRobotEnv(Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self,
                 render_mode: str = None,
                 max_steps: int = 5000,
                 screen_size_x: int = 1024,
                 screen_size_y: int = 768,
                 grid_config: GridConfig = GridConfig(),
                 obs_config: ObservationConfig = ObservationConfig(),
                 mission_space: MissionSpace = MissionSpace()):

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.screen_size_x = screen_size_x
        self.screen_size_y = screen_size_y
        self.clock = None
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))
        self.reward = 0
        self.tmp_count = 0
        self.counts = {}

        self.mission = ""

        self.obs = Observation(grid_config=grid_config, config=obs_config)
        print(self.obs.get_grid())

        cloud_space = Box(
            low=0,
            high=255,
            shape=(7, 7, 3),
            dtype="int64",
        )

        self.observation_space = Dict(
            {
                "image": cloud_space,
   #             "direction": spaces.Discrete(4),
                "mission": mission_space
            }
        )
        self.step_count = 0
        self.window = None

    def step(self, action):

        # print(f"SC: [{self.step_count}]")
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # if action == self.actions.left:
        #     self.obs.agent_dir -= 1
        #     if self.obs.agent_dir < 0:
        #         self.obs.agent_dir += 4
        # elif action == self.actions.right:
        #     self.obs.agent_dir = (self.obs.agent_dir + 1) % 4
        # elif action == self.actions.move:
        #     self.reward += self.obs.move()
        if action == self.actions.forward:
            self.reward += self.obs.move((1, 0))
        elif action == self.actions.backward:
            self.reward += self.obs.move((-1, 0))
        elif action == self.actions.left:
            self.reward += self.obs.move((0, 1))
        elif action == self.actions.right:
            self.reward += self.obs.move((0, -1))
        elif action == self.actions.dose:
            self.reward += self.obs.dose()
        # elif action == self.actions.done:
        #    pass
        else:
            raise ValueError(f"Unknown action: {action}")

        #corner = self.obs.grid.get_start_point(6, 6)
        #print(corner)
        #print(self.obs.agent_pos)
        #print("________________-")
        if self.step_count >= self.max_steps or self.obs.grid.is_healed():
            truncated = True
            reward = self._reward()

        #print(self.step_count)
        #print(action)
        #print(self.obs.grid._infected_cells_count)
        obs = {
            "image": self.obs.get_grid(),
    #        "direction": self.obs.agent_dir,
            "mission": self.mission
        }

        # if self.render_mode == "human":
        #     self.render()

        # Tuple based on which we index the counts
        # We use the position after an update
        tup = (self.obs.agent_pos[0], self.obs.agent_pos[1])

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / (new_count**2)
        reward += bonus

       # print(f"STEP: {self.step_count}, rew: {reward}")
        return obs, reward, terminated, truncated, {}

    def render(self, num = 1):
        v, c = self.obs.grid.render_encode()
        c[self.obs.agent_pos[0]][self.obs.agent_pos[1]][self.obs.agent_pos[2]] = "#111111"
        w = Window(v, c)
        w.save_plot(self.step_count,num)

    def reset(self,
              *,
              seed: int | None = None,
              options: dict | None = None) -> tuple[ObsType, dict]:
        self.obs.reset()
        self.step_count = 0
        self.reward = 0

        obs = {
            "image": self.obs.get_grid(),
     #       "direction": self.obs.agent_dir,
            "mission": self.mission
        }

        return obs, {}

    def _reward(self) -> float:
        return 1 - 0.9 * (self.step_count / self.max_steps)
