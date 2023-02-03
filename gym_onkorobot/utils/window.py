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
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title: str):
        self.title = title
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        X, Y, Z = axes3d.get_test_data(0.05)
        self.ax.plot_surface(
            X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
        )
        self.closed = False

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        plt.close()
        self.closed = True


if __name__ == "__main__":
    w = Window("test")
    w.show(block=True)
    w.close()
