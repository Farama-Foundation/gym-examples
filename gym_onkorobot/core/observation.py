import numpy as np
import random

from gym_onkorobot.core.grid import Grid
from gym_onkorobot.core.configs import GridConfig, ObservationConfig
from gym_onkorobot.utils.point import Point


class Observation:
    def __init__(self,
                 config: ObservationConfig = ObservationConfig(),
                 grid_config: GridConfig = GridConfig()):
        self.c = config
        self.grid = Grid(grid_config)

        self.reset()

    def move(self, orientation):
        nxt = [x+y for x, y in zip(self.agent_pos, orientation)]
        if self.grid.is_in_borders(Point(*nxt)): #TODO and in surface
            self.agent_pos = nxt

    def dose(self):
        return self.grid.dose(tuple(self.agent_pos), self.c.DOSE_POWER)

    def reset(self):
        self.agent_pos = self.c.AGENT_START_POS
        self.grid.reset()

    def get_grid(self):
        return np.asarray(self.grid.encode())
