from gym_onkorobot.core.generators import plane_generator
from gym_onkorobot.utils.utils import generate_infection
from gym_onkorobot.core.grid import Grid
from gym_onkorobot.core.observation import Observation
from gym_onkorobot.core.configs import GridConfig
from gym_onkorobot.envs.onkorobot import OnkoRobotEnv
from pprint import pprint
import sys
from gym_onkorobot.utils.window import Window
from functools import partial as fwrap


def test_grid_encoder():
    c = GridConfig(2, 2, 2)
    grid = Grid(config=c)
    # pprint(grid.encode(), stream=sys.stderr)
    # pprint(grid._infected_cells_count, stream=sys.stderr)
    # pprint(grid.is_healed(), stream=sys.stderr)


# def test_obs():
#     c = GridConfig(50, 50, 50, INFECTION_GEN=fwrap(generate_infection, 0.1), GRID_GEN=plane_generator)
#     o = Observation(grid_config=c)
#     # print(o.get_grid())
#     # pprint(o.grid._infected_cells_count, stream=sys.stderr)
#     # pprint(o.grid.is_healed(), stream=sys.stderr)
#     o.dose()
#     o.dose()
#     o.dose()
#     print(o.get_grid())
#     pprint(o.grid._infected_cells_count, stream=sys.stderr)
#     pprint(o.grid.is_healed(), stream=sys.stderr)
#
#     v, c = o.grid.render_encode()
#    # pprint(v)
#    # pprint(c)
#     w = Window(v, c)
#     w.imshow()


def test_vis():
    c = GridConfig(15, 15, 15, INFECTION_GEN=fwrap(generate_infection, 0.2), GRID_GEN=plane_generator)
    env = OnkoRobotEnv(grid_config=c)
    env.render()
    # episodes = 20
    # for episode in range(1, episodes + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0
    #     count = 0
    #     while not done:
    #         action = env.action_space.sample()
    #         n_state, reward, done, term, info = env.step(action)
    #         score += reward
    #     print('Episode:{} Score:{}'.format(episode, score))