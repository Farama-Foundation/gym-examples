# Gym-onkorobot

### Description
    Solve the problem of dose determination with reinforcement learning.

### Action Space

The action is a `ndarray` with shape `(4,)`. it looks like: `{x, y, z, d}`.
   
| NUM | Action                                    | Type         | Min  | Max |
|-----|-------------------------------------------|--------------|------|-----|
| X   | Step on the X-axis from the current state | Сontinuously | -Inf | Inf |
| Y   | Step on the y-axis from the current state | Сontinuously | -Inf | Inf |
| Z   | Step on the Z-axis from the current state | Сontinuously | -Inf | Inf |
| D   | Dose to current point                     | Discrete     | 0    | 1   |


### Observation Space

The observation is an array with shape `(X, Y, Z, K)` with the values corresponding to the following positions and velocities:

| Num | Observation   | Min | Max |
|-----|---------------|-----|-----|
| X   | Cart Position | 0   | Inf |
| Y   | Cart Velocity | 0   | Inf |
| Z   | Pole Angle    | 0   | Inf |
| K   | Is tumor      | 0   | 1   |

### Variables in our task

- X0, Y0, Z0 - Coordinate of manipulator
- Angle
- Distance
- Radius
- Time

### Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
including the termination step, is allotted. The threshold for rewards is 475 for v1.

### Starting State

All observations are assigned a uniformly random value in `(-0.05, 0.05)`

### Episode End

The episode ends if any one of the following occurs:
1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)

### Arguments

```python
gymnasium.make('gym_onkorobot/OnkoRobot-v0')
```

No additional arguments are currently supported.
