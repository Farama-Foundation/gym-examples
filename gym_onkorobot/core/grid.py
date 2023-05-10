from gym_onkorobot.utils.voxel import Voxel
from gym_onkorobot.utils.point import Point
from gym_onkorobot.core.configs import GridConfig, RenderConfig

from typing import Callable
from functools import partial as fwrap
from dataclasses import astuple

import random
import numpy as np
from matplotlib.colors import to_hex


class Grid:
    def __init__(self,
                 config: GridConfig = GridConfig(),
                 render_config: RenderConfig = RenderConfig()):

        self.c = config
        self.rc = render_config
        self._grid, self._infected_cells_count = self.c.GRID_GEN()
        self._reward = 1.0 / self._infected_cells_count

    def encode(self):
        """Для нейросети нужно представление в виде массива"""

        grid = []
        for i in range(self.c.X_SHAPE):
            grid.append([])
            for j in range(self.c.Y_SHAPE):
                grid[i].append([])
                for k in range(self.c.Z_SHAPE):  # Уровень облученности, Флаг зараженности
                    p = self._grid[(i, j, k)]
                    grid[i][j].append(astuple(p))
        return grid

    def render_encode(self):
        """Для нейросети нужно представление в виде массива"""

        grid = []
        colors = []
        for i in range(self.c.X_SHAPE):
            grid.append([])
            colors.append([])
            for j in range(self.c.Y_SHAPE):
                grid[i].append([None for l in range(self.c.Z_SHAPE)])
                colors[i].append([None for l in range(self.c.Z_SHAPE)])
                for k in range(self.c.Z_SHAPE):  # Уровень облученности, Флаг зараженности
                    p = self._grid[(i, j, k)]
                    bc = bool(p.is_body_cell)
                    if bc:
                        colors[i][j][k] = self.rc.BODY_COLOR
                        grid[i][j][k] = True
                    #TODO float
                    if bool(p.is_infected):
                        colors[i][j][k] = "#FF0000"
                    if p.exposure_level == p.is_infected:
                        colors[i][j][k] = "#0000FF"
        return np.asarray(grid), np.asarray(colors)

    def is_healed(self):
        return not bool(self._infected_cells_count)

    def is_in_borders(self, p: Point):
        return True if 0 <= p.x < self.c.X_SHAPE and 0 <= p.y < self.c.Y_SHAPE and 0 <= p.z < self.c.Z_SHAPE else False

    def is_body_cell(self, p: Point):
        return bool(self._grid[astuple(p)].is_body_cell)

    def get_start_point(self):
        sp = None
        count = 0
        for i in range(self.c.Z_SHAPE):
            p = Point(0, 0, i)
            if self.is_body_cell(p):
                count += 1
               # print(f"START: {p}")
                sp = astuple(p)
        #print(f"COUNT: {count}")
        #print(f"START POINT: {sp}")
        return sp

    def dose(self, p: tuple, power: int):
        reward = 0
        dosed = bool(self._grid[p].exposure_level)
        if not dosed:
            self._grid[p].exposure_level += power
            self._infected_cells_count -= 1
            reward = self._reward
        return reward

    def decode(self):
        pass

    def reset(self):
        self._grid, self._infected_cells_count = self.c.GRID_GEN()
        self._reward = 1.0 / self._infected_cells_count
