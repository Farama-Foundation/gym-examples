from dataclasses import dataclass, field
from pandera import Column
from typing import Callable
from functools import partial as fwrap
from gym_onkorobot.core.generators import plane_generator

import random
import pandera as pa


@dataclass(frozen=False)
class GridConfig:
    X_SHAPE: int = 10
    Y_SHAPE: int = 10
    Z_SHAPE: int = 10
    SURFACE_GEN: Callable = fwrap(random.randint, 0, 1)
    INFECTION_GEN: Callable = fwrap(random.randint, 0, 1)
    GRID_GEN: Callable = plane_generator

    def __post_init__(self):
        self.GRID_GEN = fwrap(self.GRID_GEN,
                              self.X_SHAPE,
                              self.Y_SHAPE,
                              self.Z_SHAPE,
                              self.SURFACE_GEN,
                              self.INFECTION_GEN)


@dataclass(frozen=True)
class ObservationConfig:
    DOSE_POWER: int = 1
    AGENT_START_POS: tuple[int] = (0, 0, 0)


@dataclass(frozen=True)
class RenderConfig:
    BODY_COLOR: str = "#E3E3E3"
