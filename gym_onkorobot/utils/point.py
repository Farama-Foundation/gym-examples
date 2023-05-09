from dataclasses import dataclass
import numpy as np
from random import randint


@dataclass
class Point:
    x: int
    y: int
    z: int


def gen_plane_points(x_size: int,
                      y_size: int,
                      z_size: int,
                      z_padding_factor: float = 0.4):
    z_down = int(z_size * z_padding_factor)
    z_up = int(z_size * (1.0 - z_padding_factor))

    p1 = np.asarray([0,
                     randint(0, y_size),
                     randint(z_down, z_up)])
    p2 = np.asarray([randint(0, x_size),
                     0,
                     randint(z_down, z_up)])
    p3 = np.asarray([x_size,
                     y_size,
                     randint(z_down, z_up)])
    return p1, p2, p3