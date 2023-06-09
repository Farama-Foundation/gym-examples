import numpy as np
import random

from gym_onkorobot.core.grid import Grid
from gym_onkorobot.core.configs import GridConfig, ObservationConfig, RenderConfig
from gym_onkorobot.utils.point import Point


class Observation:
    def __init__(self,
                 config: ObservationConfig = ObservationConfig(),
                 grid_config: GridConfig = GridConfig(),
                 render_config: RenderConfig = RenderConfig()):
        self.c = config
        self.grid = Grid(config=grid_config, render_config=render_config)

        self.reset()

    def move_penalti(self):
        return -1.0 / ((self.grid.c.X_SHAPE ** 2 * 10))

    def move(self, orientation):
        reward = 0
        out_of_borders = True
        # if self.agent_dir == 0:
        #     orientation = (1, 0)
        # elif self.agent_dir == 1:
        #     orientation = (0, 1)
        # elif self.agent_dir == 2:
        #     orientation = (-1, 0)
        # else: #self.agent_dir == 3:
        #     orientation = (0, -1)
        movable = [(*orientation, 0), (*orientation, 1), (*orientation, -1)]
        for nxt_ori in movable:
            nxt = [x+y for x, y in zip(self.agent_pos, nxt_ori)]
            p = Point(*nxt)
            if self.grid.is_in_borders(p):
                out_of_borders = False
                if self.grid.is_body_cell(p):
                    self.grid.move_agent(Point(*self.agent_pos), p)
                    self.agent_pos = nxt
                    if self.grid.is_visited(p):
                        reward += self.move_penalti()
                    break
        # if out_of_borders:
        #     reward += self.move_penalti()
        return reward

    def dose(self):
        return self.grid.dose(tuple(self.agent_pos), self.c.DOSE_POWER)

    def reset(self):
        self.grid.reset()
        self.agent_pos = self.grid.get_start_point()
        self.agent_dir = 0

    def get_grid(self):
        return self.grid.encode()
