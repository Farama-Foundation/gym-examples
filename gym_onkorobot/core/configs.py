from dataclasses import dataclass
from pandera import Column, Check
import pandera as pa


@dataclass(frozen=True)
class VoxelGridConfig:
    X_NORM_SHAPE: int = 2**10
    Y_NORM_SHAPE: int = 2**10
    Z_NORM_SHAPE: int = 2**8
    DF_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
        {
            "x": Column(float),
            "y": Column(float),
            "z": Column(float),
            "degree_exposure": Column(float),
            "roi": Column(bool),
            "amplifying_factor": Column(float)
        }
    )