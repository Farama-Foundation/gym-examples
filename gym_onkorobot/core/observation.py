class Observation:
    def __init__(self,
                 grid_size = 10,
                 dose_power = 1):
        self.grid_size = grid_size
        self.dose_power = dose_power

        self.reset()

    def dose(self):
        x = self.agent.pos()[0]
        y = self.agent.pos()[1]
        z = self.agent.pos()[2]
        self._grid[x][y][z][0] += self.dose_power
        delta = self._grid[x][y][z][1] - self._grid[x][y][z][0]
        # print(f"D: {delta}")
        # TODO сделать условие на 20%
        return 1 if abs(delta) == 0 else 0

    def is_healed(self):
        healed = True
        for i in range(self.grid_size):
          for j in range(self.grid_size):
            for k in range(self.grid_size):
              delta = self._grid[i][j][k][1] - self._grid[i][j][k][0]
              if delta > 0:
                healed = False
                return healed
        return healed


    def reset(self):
        self.agent = Laser()
        self._grid = gen_obs(self.grid_size)

    def get_grid(self):
        return np.asarray(self._grid)