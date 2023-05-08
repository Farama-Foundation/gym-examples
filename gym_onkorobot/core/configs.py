from dataclasses import dataclass, field
from pandera import Column
from typing import Callable
from functools import partial as fwrap

import random
import pandera as pa


@dataclass(frozen=True)
class GridConfig:
    X_SHAPE: int = 10
    Y_SHAPE: int = 10
    Z_SHAPE: int = 10
    SURFACE_GEN: Callable = fwrap(random.randint, 0, 1)
    INFECTION_GEN: Callable = fwrap(random.randint, 0, 1)


@dataclass(frozen=True)
class ObservationConfig:
    DOSE_POWER: int = 1
    AGENT_START_POS: tuple[int] = (0, 0, 0)
