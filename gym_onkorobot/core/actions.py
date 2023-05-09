from enum import IntEnum


class Actions(IntEnum):
    # Turn left, turn right, move forward, turn up, turn down
    left = 0
    right = 1
    forward = 2
    backward = 3
    upforward = 4
    upbackward = 5
    upleft = 6
    upright = 7
    downforward = 8
    downbackward = 9
    downleft = 10
    downright = 11
    dose = 12
