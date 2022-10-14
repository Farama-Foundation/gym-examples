"""This file defines functions to handle dict observation spaces."""

import numpy as np


def single_obs(obs_dict):
    """Convert an observation dictionary to a concatenated vector.

    Args:
        obs_dict: {'observation', 'achieved_goal', 'desired_goal'} dictionary

    Returns:
        vec [observation, achieved_goal, desired_goal]
    """
    return np.concatenate((
        obs_dict["observation"],
        obs_dict["desired_goal"]),  axis=-1)
