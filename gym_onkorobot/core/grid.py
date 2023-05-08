from gym_onkorobot.utils.voxel import Voxel
from gym_onkorobot.utils.point import Point
from gym_onkorobot.core.configs import GridConfig

from typing import Callable
from functools import partial as fwrap
from dataclasses import astuple

import random
import numpy as np


def grid_generator(x: int,
                   y: int,
                   z: int,
                   surface_gen: Callable,
                   infection_gen: Callable) -> tuple[dict, int]:
    grid = dict()
    count = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):  # Уровень облученности, Флаг зараженности
                p = (i, j, k)
                infected = infection_gen()
                v = Voxel(exposure_level=0,
                          is_infected=infected)
                if infected != 0:
                    count += 1
                grid[p] = v
    return grid, count


class Grid:
    def __init__(self,
                 config: GridConfig = GridConfig()):

        self.c = config
        self._grid, self._infected_cells_count = grid_generator(self.c.X_SHAPE, self.c.Y_SHAPE, self.c.Z_SHAPE,
                                                                self.c.SURFACE_GEN, self.c.INFECTION_GEN)

    def encode(self):
        """Для нейросети нужно представление в виде массива"""

        grid = []
        for i in range(self.c.X_SHAPE):
            grid.append([])
            for j in range(self.c.Y_SHAPE):
                grid[i].append([])
                for k in range(self.c.Z_SHAPE):  # Уровень облученности, Флаг зараженности
                    grid[i][j].append(astuple(self._grid[(i, j, k)]))
        return grid

    def is_healed(self):
        return not bool(self._infected_cells_count)

    def is_in_borders(self, p: Point):
        return True if 0 <= p.x < self.c.X_SHAPE and 0 <= p.y < self.c.Y_SHAPE and 0 <= p.z < self.c.Z_SHAPE else False

    def dose(self, p: tuple, power: int):
        self._grid[p].exposure_level += power
        delta = self._grid[p].exposure_level - self._grid[p].is_infected
        reward = 0
        # TODO сделать условие на 20%
        if abs(delta) == 0:
            reward = 1
            self._infected_cells_count -= 1
        return reward

    def decode(self):
        pass

    def reset(self):
        self._grid, self._infected_cells_count = grid_generator(self.c.X_SHAPE, self.c.Y_SHAPE, self.c.Z_SHAPE,
                                    self.c.SURFACE_GEN, self.c.INFECTION_GEN)