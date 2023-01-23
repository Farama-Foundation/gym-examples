# voxel grid
from typing import Tuple


class VoxelGrid:
    x_shape, y_shape, z_shape = None, None, None

    def __init__(self, shape: Tuple[int, int, int]):
        self.x_shape, self.y_shape, self.z_shape = shape


    # def update