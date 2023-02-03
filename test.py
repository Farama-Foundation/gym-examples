import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import matplotlib.pyplot as plt


def generate_pc(n: int = 10):
    rng = np.random.default_rng()
    coordinates = pd.DataFrame(10 * rng.random((n, 3)) - 10, columns=['x', 'y', 'z'])
    print(coordinates.head())
    print(list(coordinates.columns))
    return coordinates

coordinates = generate_pc()
tiny_cloud = PyntCloud(coordinates)


# проверка правильности вокселизации "на глаз"
# исходный фрейм
coordinates.plot.scatter(x="x", y="y")

# воксель-сетка
voxelgrid_id = tiny_cloud.add_structure("voxelgrid", size_x=0.2, size_y=0.2, size_z=0.2)
voxelgrid = tiny_cloud.structures[voxelgrid_id]

x = []
y = []
z = []
for idx in range(0, len(voxelgrid.voxel_n)):
    curr_number = voxelgrid.voxel_n[idx]
    voxel_center = voxelgrid.voxel_centers[curr_number]
    x.append(voxel_center[0])
    y.append(voxel_center[1])
    z.append(voxel_center[2])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)
plt.show()