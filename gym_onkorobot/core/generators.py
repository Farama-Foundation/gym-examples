from gym_onkorobot.utils.voxel import Voxel
from gym_onkorobot.utils.point import gen_plane_points

from typing import Callable
from dataclasses import astuple
import numpy as np
from random import random


def distance_from_plane(plane: tuple, point: tuple):
    # print(plane)
    # print(point)
    return np.fabs((point[0] * plane[0] + point[1] * plane[1] + point[2] * plane[2] + plane[3])) / np.sqrt(
        plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)


def plane_equation(p1, p2, p3):
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p2[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * p1[0] - b * p1[1] - c * p1[2])
    return a, b, c, d


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
                body_cell = surface_gen()
                infected = 0
                if body_cell:
                    infected = infection_gen()
                    if infected != 0:
                        count += 1
                v = Voxel(exposure_level=0,
                          is_infected=infected,
                          is_body_cell=body_cell)
                grid[p] = v
    return grid, count


def plane_generator(x: int,
                   y: int,
                   z: int,
                   surface_gen: Callable,
                   infection_gen: Callable) -> tuple[dict, int]:
    grid = dict()
    count = 0
    p1, p2, p3 = gen_plane_points(x, y, z)
    a, b, c, d = plane_equation(p1, p2, p3)

    for i in range(x):
        for j in range(y):
            min_point = (i, j, 0)
            min_dist = 10
            for k in range(z):  # Уровень облученности, Флаг зараженности
                p = (i, j, k)
                body_cell = False
                infected = False
                #if abs(np.dot(cp, p) - d) <= eps:
                po = distance_from_plane((a, b, c, d), p)
                if po <= min_dist:
                    min_dist = po
                    min_point = p
                v = Voxel(exposure_level=0,
                          is_infected=infected,
                          is_body_cell=body_cell)
                grid[p] = v
            grid[min_point].is_body_cell = True
            if count < 6:
                grid[min_point].is_infected = infection_gen()
            if grid[min_point].is_infected:
                count += 1
                #print(f"INF: {min_point}")
    return grid, count
