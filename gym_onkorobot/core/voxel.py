from gym_onkorobot.utils.point_cl import Point


class Voxel(Point):
    degree_exposure: float  # 0..1
    roi: bool
    amplifying_factor: float  # 0..1
