import numpy as np
import open3d as o3d

# proof of concept

rng = np.random.default_rng()
n_pcd = 10 * rng.random((10, 3)) - 10
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(n_pcd)

black = 111
R = [black for i in range(10)]
G = [black for i in range(10)]
B = [black for i in range(10)]
rgb = np.asarray([R,G,B])
rgb_t = np.transpose(rgb)
pcd.colors = o3d.utility.Vector3dVector(rgb_t.astype(np.float) / 255.0)
print(np.asarray(pcd.points))
print(np.asarray(pcd.colors))

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=1)

octree = o3d.geometry.Octree(max_depth=5)
octree.create_from_voxel_grid(voxel_grid)
o3d.visualization.draw_geometries([voxel_grid], point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)

# api для "несуществующей координаты"

voxel = np.asarray([40,40,412])
print(f"FALSE OCTree coord: {octree.locate_leaf_node(voxel)}\n ______________ \n")

# проверка потери/сохранения индексации

print(f"PCloud coord: {pcd.points[0]}")
print(f"Voxelgrid coord: {voxel_grid.get_voxel(pcd.points[0])}")
print(f"OCTree coord: {octree.locate_leaf_node(voxel)}")

print("____________________")

octree_p = o3d.geometry.Octree(max_depth=5)
octree_p.convert_from_point_cloud(point_cloud=pcd, size_expand=0.1)
o3d.visualization.draw_geometries([octree_p])

print(f"PCloud coord: {pcd.points[0]}")
print(f"OCTree coord: {octree.locate_leaf_node(pcd.points[0])}")
