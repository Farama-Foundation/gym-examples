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


def test_shape():
    data = [
        {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "degree_exposure": 0.4,
            "roi": True,
            "amplifying_factor": 0.01
        },
        {
            "x": 2.0,
            "y": 5.0,
            "z": 1.0,
            "degree_exposure": 0.3,
            "roi": False,
            "amplifying_factor": 0.01
        }
    ]
    grid = VoxelGrid(dataframe=DataFrame.from_dict(data))
    assert grid.shape() == (1000, 3)