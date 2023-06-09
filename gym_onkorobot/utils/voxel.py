from dataclasses import dataclass


@dataclass
class Voxel:
    #bedrock: int = 0
    is_infected: int
    exposure_level: int # TODO float
    # exposure_factor: float
    is_body_cell: int
    is_visited: bool = False
    agent: int = 0
