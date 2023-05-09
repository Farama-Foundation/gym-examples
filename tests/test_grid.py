from gym_onkorobot.core.generators import plane_generator, grid_generator
from gym_onkorobot.core.grid import Grid
from gym_onkorobot.core.observation import Observation
from gym_onkorobot.core.configs import GridConfig
from pprint import pprint
import sys
from gym_onkorobot.utils.window import Window


def test_grid_encoder():
    c = GridConfig(2, 2, 2)
    grid = Grid(config=c)
    pprint(grid.encode(), stream=sys.stderr)
    pprint(grid._infected_cells_count, stream=sys.stderr)
    pprint(grid.is_healed(), stream=sys.stderr)


def test_obs():
    c = GridConfig(20, 20, 20, GRID_GEN=plane_generator)
    o = Observation(grid_config=c)
    # print(o.get_grid())
    # pprint(o.grid._infected_cells_count, stream=sys.stderr)
    # pprint(o.grid.is_healed(), stream=sys.stderr)
    o.dose()
    o.dose()
    o.dose()
    print(o.get_grid())
    pprint(o.grid._infected_cells_count, stream=sys.stderr)
    pprint(o.grid.is_healed(), stream=sys.stderr)

    v, c = o.grid.render_encode()
   # pprint(v)
   # pprint(c)
    w = Window(v, c)
    w.imshow()
