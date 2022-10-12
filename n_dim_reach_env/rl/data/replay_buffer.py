from typing import Optional, Union

import gym
import gym.spaces
import numpy as np

from human_robot_gym.rl.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    """The replay buffer for off-policy RL.
    Args:
        observation_space (gym.Space): The observation space of the environment
        action_space (gym.Space): The action space of the environment
        capacity (int): Number of transitions in the buffer
        next_observation_space (gym.Space, Optional): The next observation space of the environment.
            If omitted, will be the same as observation_space.
        achieved_goal_space (gym.Space, Optional): The achieved goal space of the environment.
            Only to be used for HER / GoalEnv.
        desired_goal_space (gym.Space, Optional): The desired goal space of the environment.
            Only to be used for HER / GoalEnv.
    """
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 next_observation_space: Optional[gym.Space] = None,
                 achieved_goal_space: Optional[gym.Space] = None,
                 desired_goal_space: Optional[gym.Space] = None):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space,
                                                  capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
        )
        if desired_goal_space is None and achieved_goal_space is not None:
            desired_goal_space = achieved_goal_space
        if achieved_goal_space is not None:
            dataset_dict['achieved_goals'] = np.empty((capacity, *achieved_goal_space.shape),
                                                       dtype=achieved_goal_space.dtype)
            dataset_dict['desired_goals'] = np.empty((capacity, *desired_goal_space.shape),
                                                      dtype=desired_goal_space.dtype)
            dataset_dict['next_achieved_goals'] = np.empty((capacity, *achieved_goal_space.shape),
                                                            dtype=achieved_goal_space.dtype)
            dataset_dict['next_desired_goals'] = np.empty((capacity, *desired_goal_space.shape),
                                                           dtype=desired_goal_space.dtype)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
