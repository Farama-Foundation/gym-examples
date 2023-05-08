from gym_onkorobot.core.grid import grid_generator
from gym_onkorobot.core.grid import Grid
from gym_onkorobot.core.observation import Observation
from gym_onkorobot.core.configs import GridConfig
from pandas import DataFrame
from pandera.errors import SchemaError
import numpy as np
from pprint import pprint
import sys


def test_grid_encoder():
    c = GridConfig(2, 2, 2)
    grid = Grid(config=c)
    pprint(grid.encode(), stream=sys.stderr)
    pprint(grid._infected_cells_count, stream=sys.stderr)
    pprint(grid.is_healed(), stream=sys.stderr)


def test_obs():
    c = GridConfig(2, 2, 2)
    o = Observation(grid_config=c)
    print(o.get_grid())
    pprint(o.grid._infected_cells_count, stream=sys.stderr)
    pprint(o.grid.is_healed(), stream=sys.stderr)
    o.dose()
    o.dose()
    o.dose()
    print(o.get_grid())
    pprint(o.grid._infected_cells_count, stream=sys.stderr)
    pprint(o.grid.is_healed(), stream=sys.stderr)