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
        self._grid[self.get_start_point()].agent = 1
        self.agent_pos = self.get_start_point()
        if self._infected_cells_count == 0:
            self._reward = 1.0
        else:
            self._reward = 1.0 / self._infected_cells_count

    def encode(self):
        """Для нейросети нужно представление в виде массива"""

        view_radius = 3
        d = view_radius*2 + 1
        x_min = self.agent_pos[0] - view_radius - 1
        x_max = self.agent_pos[0] + view_radius
        y_min = self.agent_pos[1] - view_radius - 1
        y_max = self.agent_pos[1] + view_radius

        grid = np.zeros((d, d, 3), dtype="uint8")
        for i in range(d):
            for j in range(d):
                grid[i, j, 0] = 1

        for i, iv in zip(range(x_min, x_max), range(d)):
            for j, jv in zip(range(y_min, y_max), range(d)):
                for k in range(self.c.Z_SHAPE):  # Уровень облученности, Флаг зараженности
                    if i >= 0 and j >= 0 and i < self.c.X_SHAPE and j < self.c.Y_SHAPE:
                        p = self._grid[(i, j, k)]
                        if bool(p.is_body_cell):
                            grid[iv, jv, 0] = 0
                            #grid[iv, jv, 1] = p.agent
                            grid[iv, jv, 1] = p.is_infected
                            grid[iv, jv, 2] = int(p.is_visited)
                            #grid[i, j, 2] = p.
                            break
        #print(f"GRID: {grid}")
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
                    #    colors[i][j][k] = self.rc.BODY_COLOR
                        if p.is_visited:
                            colors[i][j][k] = self.rc.VISITED_BODY_COLOR
                        grid[i][j][k] = True
                    #TODO float
                    #if bool(p.is_infected):
                        #colors[i][j][k] = "#FF0000"
                        #if p.is_visited:
                        #    colors[i][j][k] = "#FF3333"
                    #if p.exposure_level == p.is_infected and p.is_infected == 1:
                        #colors[i][j][k] = "#0000FF"
                        #if p.is_visited:
                         #   colors[i][j][k] = "#3333FF"
        return np.asarray(grid), np.asarray(colors)

    def is_healed(self):
        return self._infected_cells_count == 0

    def is_in_borders(self, p: Point):
        return True if 0 <= p.x < self.c.X_SHAPE and 0 <= p.y < self.c.Y_SHAPE and 0 <= p.z < self.c.Z_SHAPE else False

    def is_body_cell(self, p: Point):
        return bool(self._grid[astuple(p)].is_body_cell)

    def is_visited(self, p: Point):
        return self._grid[astuple(p)].is_visited

    def set_visited(self, p: Point):
        self._grid[astuple(p)].is_visited = True

    def move_agent(self, src: Point, dst: Point):
        self._grid[astuple(src)].agent = 0
        self.set_visited(src)
        self._grid[astuple(dst)].agent = 1
        self.agent_pos = astuple(dst)
       # print(f"AGENT: {self._grid[astuple(dst)]}")
        #print(f"AGENTCC: {dst}")

    def get_start_point(self, x = 0, y = 0):
        sp = None
        count = 0
        for i in range(self.c.Z_SHAPE):
            p = Point(x, y, i)
            if self.is_body_cell(p):
                count += 1
               # print(f"START: {p}")
                sp = astuple(p)
        #print(f"COUNT: {count}")
        #print(f"START POINT: {sp}")
        return sp

    def dose(self, p: tuple, power: int):
        reward = 0
        if self._grid[p].is_infected:
            reward = self._reward
            self._infected_cells_count -= 1
            self._grid[p].is_infected = 0
        # infected = bool(self._grid[p].is_infected)
        # #print("DOSE")
        # if dose == 0:
        #     self._grid[p].exposure_level += power
        # if infected:
        #     #print(self._grid[p].exposure_level)
        #     #self._grid[p].is_infected = 0
        #     self._infected_cells_count -= 1
        return reward

    def decode(self):
        pass

    def reset(self):
        self._grid, self._infected_cells_count = self.c.GRID_GEN()
        self._grid[self.get_start_point()].agent = 1
        self.agent_pos = self.get_start_point()
        if self._infected_cells_count == 0:
            self._reward = 1.0
        else:
            self._reward = 1.0 / self._infected_cells_count
