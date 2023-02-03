# voxel grid
from typing import Tuple
from pandas import DataFrame
from pyntcloud import PyntCloud

from gym_onkorobot.core.configs import VoxelGridConfig as c


class VoxelGrid:
    x_shape, y_shape, z_shape = None, None, None

    def __init__(self,
                 dataframe: DataFrame,
                 shape: Tuple[int, int, int] = (c.X_NORM_SHAPE, c.Y_NORM_SHAPE, c.Z_NORM_SHAPE),
                 ):
        self.x_shape, self.y_shape, self.z_shape = shape
        self.pc = None
        self.voxelgrid = None
        self._create_grid(cloud=dataframe)

    @staticmethod
    def validate_df(df):
        schema = c.DF_SCHEMA
        schema.validate(df)

    def _create_grid(self, cloud):
        """Method for standardization of point cloud to same format for RL simulator"""
        VoxelGrid.validate_df(cloud)
        self.pc = PyntCloud(cloud)

        voxelgrid_id = self.pc.add_structure("voxelgrid", n_x=self.x_shape, n_y=self.y_shape, n_z=self.z_shape)
        self.voxelgrid = self.pc.structures[voxelgrid_id]

    def shape(self):
        """Returns shape of voxel grid"""
        return self.voxelgrid.voxel_centers.shape

    def forward(self, p0: Tuple[float, float, float], dir: str):
        #TODO get next point
        pass