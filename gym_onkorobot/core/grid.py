# voxel grid
from typing import Tuple
from pandas import DataFrame

from gym_onkorobot.core.configs import VoxelGridConfig as c


class VoxelGrid:
    x_shape, y_shape, z_shape = None, None, None

    def __init__(self,
                 dataframe: DataFrame,
                 shape: Tuple[int, int, int] = (c.X_NORM_SHAPE, c.Y_NORM_SHAPE, c.Z_NORM_SHAPE),
                 ):
        self.x_shape, self.y_shape, self.z_shape = shape
        self.voxels_coords = None
        self._create_grid(df=dataframe)

    @staticmethod
    def validate_df(df):
        schema = c.DF_SCHEMA
        schema.validate(df)

    def _create_grid(self, df):
        """Method for standardization of point cloud to same format for RL simulator"""
        self.validate_df(df)




        pass

