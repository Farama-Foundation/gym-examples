"""This file defines functions to scale actions.

The droq implementation only outputs actions in [-1, 1].
"""

import numpy as np


def scale_action(
        action: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        squash_output: bool = True
        ) -> np.ndarray:
    """Rescale the action from [low, high] to [-1, 1].

    (no need for symmetric action space)
    :param action: Action to scale
    :param low: Lower value of actions
    :param high: Upper value of actions
    :param squash_output: Whether to squash the output to [-1, 1]
    :return: Scaled action
    """
    if squash_output:
        a = 2.0 * ((action - low) / (high - low)) - 1.0
    else:
        a = action
    return np.clip(a, -1, 1)


def unscale_action(
        scaled_action: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        squash_output: bool = True
        ) -> np.ndarray:
    """Rescale the action from [-1, 1] to [low, high].

    (no need for symmetric action space)
    :param scaled_action: Action to un-scale
    :param low: Lower value of actions
    :param high: Upper value of actions
    :param squash_output: Whether to squash the output to [low, high]
    :return: Un-scaled action
    """
    if squash_output:
        a = low + (0.5 * (scaled_action + 1.0) * (high - low))
    else:
        a = scaled_action
    return np.clip(a, low, high)
