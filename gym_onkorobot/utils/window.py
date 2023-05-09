import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, axes3d
except ImportError:
    raise ImportError(
        "To display the environment in a window, please install matplotlib, eg: `pip3 install --user matplotlib`"
    )


class Window:
    def __init__(self,
                 voxels,
                 colors):
        self.plot = plt.figure().add_subplot(projection='3d')
        plt.xlabel('xlabel')
        plt.ylabel('ylabel')
        self.voxels = voxels
        self.colors = colors

    def imshow(self, path: str = None):
        self.plot.voxels(self.voxels, facecolors=self.colors, edgecolor='k')
        if not path:
            plt.show()

    def animation(self, path: str = None):
        pass
