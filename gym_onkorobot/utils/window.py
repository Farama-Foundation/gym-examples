import datetime
import io
import numpy as np
from PIL import Image
import pygame

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
        self.fig = plt.figure()
        self.plot = self.fig.add_subplot(projection='3d')
        plt.xlabel('xlabel')
        plt.ylabel('ylabel')
        self.voxels = voxels
        self.colors = colors

    def save_plot(self, step, num):
        self.plot.voxels(self.voxels, facecolors=self.colors, edgecolor='k')
        prefix = ""
        if step >= 100:
            prefix = str(step)
        elif 100 > step >= 10:
            prefix = f"0{step}"
        else:
            prefix = f"00{step}"
        plt.savefig(f"/home/owlengineer/git/gym-onkorobot/imgs{num}/step{prefix}.png", format='png')
