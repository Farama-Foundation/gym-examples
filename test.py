import numpy as np
from pyntcloud import PyntCloud
import pandas as pd


def generate_pc(n: int = 100):
    rng = np.random.default_rng()
    coordinates = pd.DataFrame(100 * rng.random((n, 3)) - 100, columns=['x', 'y', 'z'])
    print(coordinates.head())
    print(list(coordinates.columns))
    return coordinates

coordinates = generate_pc()
tiny_cloud = PyntCloud(coordinates)
# voxel
print(tiny_cloud)
