from dataclasses import dataclass
from pandera import Column
import pandera as pa


@dataclass(frozen=True)
class VoxelGridConfig:
    X_NORM_SHAPE: int = 10
    Y_NORM_SHAPE: int = 10
    Z_NORM_SHAPE: int = 10
    DF_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema( {
            "x": Column(float),
            "y": Column(float),
            "z": Column(float),
            "degree_exposure": Column(float),
            "roi": Column(bool),
            "amplifying_factor": Column(float)
        } )
    VOXEL_SIZE: float = 0.01
