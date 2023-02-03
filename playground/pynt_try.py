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


def check_voxelization():
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


def check_field_adding():
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
    df = pd.DataFrame.from_dict(data)
    tiny_cloud = PyntCloud(df)
    print(tiny_cloud.points["roi"][0])

    #voxelgrid_id = tiny_cloud.add_structure("voxelgrid", size_x=0.1, size_y=0.1, size_z=0.1)
    voxelgrid_id = tiny_cloud.add_structure("voxelgrid", n_x=10, n_y=10, n_z=10)
    voxelgrid = tiny_cloud.structures[voxelgrid_id]
    print(voxelgrid.voxel_centers.shape)


check_field_adding()