from typing import Optional, Union, List, Tuple

import gymnasium as gym
#from gymnasium import spaces
#import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType, Env
from gymnasium.spaces import Box, Discrete, Dict, Space
#from gym_onkorobot.core.actions import Actions
#from gym_onkorobot.core.grid import VoxelGrid
#from gym_onkorobot.utils.window import Window
from typing import Any
import random

from gymnasium.utils.env_checker import check_env

from typing import Any, Callable

from gymnasium import spaces
from gymnasium.utils import seeding

from gym_onkorobot.core.actions import Actions

def gen_obs(size):
  data = []
  for i in range(size):
    data.append([])
    for j in range(size):
      data[i].append([])
      for k in range(size): # Уровень облученности, Флаг зараженности
        data[i][j].append([0, random.randint(0,1)])
  #print(data)
  return np.asarray(data)


class Laser():
    def __init__(self,
                 start_pos = [0,0,0]):
        self._agent_pos = start_pos

    def move(self, orientation):
        res = [x+y for x, y in zip(self._agent_pos, orientation)]
        #TODO перенести метод в OBS
        if res[0] < 10 and res[1] < 10 and res[2] < 10 and res[0] >= 0 and res[1] >= 0 and res[2] >= 0:
            self._agent_pos = res

    def pos(self):
        return self._agent_pos

    def reset(self, max_size):
        self._agent_pos = [0,0,0]


class BabyAIMissionSpace(Space[str]):
    def __init__(self):
        pass

    @staticmethod
    def _gen_mission():
        return "Heal all infected points."

    def contains(self, x: str):
        return True


class OnkoRobotEnv(Env):
    def __init__(self,
                 mission_space=BabyAIMissionSpace()):

        self.x_size = 10
        self.y_size = 10
        self.z_size = 10

        self.max_steps = 1000
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))

        self.mission = ""

        cloud_space = Box(
            low=0,
            high=255,
            shape=(self.x_size, self.y_size, self.z_size, 2),
            #shape=(1, self.x_size, self.y_size, self.z_size, 2),
            dtype="uint32",
        )

        self.obs = Observation()
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

        if action == self.actions.down:
            self.obs.agent.move([0, 0, -1])
        elif action == self.actions.up:
            self.obs.agent.move([0, 0, 1])
        elif action == self.actions.left:
            self.obs.agent.move([0, -1, 0])
        elif action == self.actions.right:
            self.obs.agent.move([0, 1, 0])
        elif action == self.actions.forward:
            self.obs.agent.move([-1, 0, 0])
        elif action == self.actions.backward:
            self.obs.agent.move([1, 0, 0])
        elif action == self.actions.dose:
            reward = self.obs.dose()
        # elif action == self.actions.done:
        #    pass
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.obs.is_healed() or self.step_count >= self.max_steps:
            truncated = True

        obs = {
            "image": self.obs.get_grid(),
            "mission": self.mission
        }

       # print(f"STEP: {self.step_count}, rew: {reward}")
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

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
