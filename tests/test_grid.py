from gym_onkorobot.core.grid import VoxelGrid
from pandas import DataFrame
from pandera.errors import SchemaError
import numpy as np


def test_validation():
    res = False
    rng = np.random.default_rng()
    coordinates = DataFrame(10 * rng.random((5, 3)) - 10, columns=['x', 'y', 'z'])
    try:
        VoxelGrid.validate_df(coordinates)
    except SchemaError:
        res = True
    assert res

