from gym_onkorobot.utils.point import Point
from dataclasses import dataclass


@dataclass
class Voxel:
    exposure_level: int # TODO float
    # exposure_factor: float
    is_infected: int
    #is_body_cell: int