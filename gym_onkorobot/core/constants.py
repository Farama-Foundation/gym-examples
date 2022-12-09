import numpy as np

# Map of state names to integers
STATE_TO_IDX = {"sore": 0, "healthy": 1}
# Map of agent direction indices to vectors
# TODO
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]
