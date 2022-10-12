from human_robot_gym.rl.data.replay_buffer import ReplayBuffer

from typing import Dict, Iterable, Optional, Union, Any
from enum import Enum
from collections import deque

import gym
import gym.spaces
import numpy as np
from flax.core import frozen_dict

from human_robot_gym.rl.data.dataset import DatasetDict, _sample


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
}


def get_time_limit(env: gym.GoalEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    Args:
        env: Environment from which we want to get the time limit.
        current_max_episode_length: Current value for max_episode_length.
    Returns:
        max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


class HEReplayBuffer(ReplayBuffer):
    """Hindsight Experience Replay Buffer.

    Samples k new transitions for every given transition with virtual goals.
    Enables sparse reward learning.

    Args:
        observation_space: gym.Space
        action_space: gym.Space
        capacity: number of real transitions (not virtual ones)
        achieved_goal_space: gym.Space of the achieved goal
        desired_goal_space (Optional, gym.Space): If omitted, achieved_goal_space will be used.
        next_observation_space: Optional (gym.space)
        max_episode_length: The maximum length of an episode. If not specified,
                            it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit``
                            wrapper.
        goal_selection_strategy: Strategy for sampling goals for replay.
                                 One of ['episode', 'final', 'future']
        n_sampled_goal: Number of virtual transitions to create per real transition,
                        by sampling new goals.
        handle_timeout_termination: Handle timeout termination (due to timelimit) separately and treat the task as
                                    infinite horizon task.
                                    https://github.com/DLR-RM/stable-baselines3/issues/284
    """
    def __init__(self,
                 env: gym.GoalEnv,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 achieved_goal_space: gym.Space,
                 desired_goal_space: Optional[gym.Space] = None,
                 next_observation_space: Optional[gym.Space] = None,
                 max_episode_length: Optional[int] = None,
                 n_sampled_goal: int = 4,
                 goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
                 handle_timeout_termination: bool = True):
        if desired_goal_space is None and achieved_goal_space is not None:
            desired_goal_space = achieved_goal_space
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         capacity=capacity,
                         next_observation_space=next_observation_space)
        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy
        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"
        self.n_sampled_goal = n_sampled_goal
        self.handle_timeout_termination = handle_timeout_termination
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, max_episode_length)
        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = capacity // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0
        self.pos = 0
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)
        self.episode_start_ids = np.zeros(self.max_episode_stored, dtype=np.int64)
        self.full = False
        # info has its own buffer because it cannot be initialized properly.
        # Store info dicts as it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]

    def insert(
            self,
            data_dict: DatasetDict,
            infos: Dict[str, Any],
            env: gym.GoalEnv
    ):
        """Insert a transition to the replay buffer.
        Increments the epsiode counter by 1 if done flag is set.
        """
        # Remove termination signals due to timeout
        done = data_dict["dones"]
        # If the environment changed the action.
        if "action" in infos:
            # Rescale the action from [low, high] to [-1, 1]
            if isinstance(env.action_space, gym.spaces.Box):
                # THIS ASSUMES TANH ACTIVATION FUNCTION FOR THE POLICY NETWORK!!!
                scaled_action = (
                    2.0
                    * (
                        (infos["action"] - env.action_space.low)
                        / (env.action_space.high - env.action_space.low)
                    )
                    - 1.0
                )
                data_dict["actions"] = np.clip(scaled_action, -1, 1)
            else:
                data_dict["actions"] = infos["action"]
        # Info buffer full handling
        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        if self.handle_timeout_termination:
            done_ = done or (1 - infos.get("TimeLimit.truncated", False))
        else:
            done_ = done
        data_dict["dones"] = done_
        super().insert(data_dict)
        # Fill info buffer
        self.info_buffer[self.pos].append(infos)
        # update current pointer
        self.current_idx += 1
        self.episode_steps += 1
        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode(env)
            self.episode_steps = 0

    def store_episode(self, env: gym.GoalEnv) -> None:
        """Increment episode counter and reset transition pointer."""
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx
        # Calculate HER samples
        self.compute_HER_samples(episode_id=self.pos,
                                 env=env,
                                 k=self.n_sampled_goal,
                                 strategy=self.goal_selection_strategy,)
        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        self.episode_start_ids[self.pos] = self._insert_index
        # reset transition pointer
        self.current_idx = 0

    def compute_HER_samples(self,
                            episode_id: int,
                            env: gym.GoalEnv,
                            k: Optional[int] = 4,
                            strategy: Optional[GoalSelectionStrategy] = GoalSelectionStrategy.FUTURE):
        """Create HER samples for the given episode ID.
        Should only be called after the episode is finished.

        Args:
            episode_id (int): The episode to sample from.
            env (gym.GoalEnv): To create the HER samples
            k (int): Number of HER transitions per normal transition.
            strategy (GoalSelectionStrategy): Type of goal selection.
        """
        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        episode_length = self.episode_lengths[episode_id]
        if strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            if episode_length < 2:
                return
            episode_length -= 1
        episode_indices = episode_id * np.ones([k*episode_length, ], dtype=np.int64)
        transitions_indices = np.repeat(np.arange(0, episode_length), k, axis=-1)
        her_indices = np.arange(0, k*episode_length)
        # get selected transitions
        buffer_indices = self.get_buffer_indices(episode_indices, transitions_indices)
        transitions = self.sample_non_frozen(batch_size=k*episode_length, indx=buffer_indices)

        # Convert info buffer to numpy array
        infos = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # sample new desired goals and relabel the transitions
            new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
            transitions["observations"]["desired_goal"][her_indices] = new_goals
            transitions["next_observations"]["desired_goal"][her_indices] = new_goals
            # Vectorized computation of the new reward
            transitions["rewards"][her_indices] = env.compute_reward(
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_observations"]["achieved_goal"][her_indices],
                # here we use the new desired goal
                transitions["observations"]["desired_goal"][her_indices],
                infos[her_indices],
            )
            # Vectorized computation of the new done values
            transitions["dones"][her_indices] = env.compute_done(
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_observations"]["achieved_goal"][her_indices],
                # here we use the new desired goal
                transitions["observations"]["desired_goal"][her_indices],
                infos[her_indices],
            )
            transitions["masks"] = transitions["dones"]
        for i in range(k*episode_length):
            transition = dict()
            for k in transitions.keys():
                if isinstance(transitions[k], dict):
                    transition[k] = _sample(transitions[k], i)
                else:
                    transition[k] = transitions[k][i]
            super().insert(transition)

    def sample(self,
               batch_size: int,
               env: gym.GoalEnv,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        return self._sample_transitions(batch_size, env=env)  # pytype: disable=bad-return-type

    def _sample_transitions(
        self,
        batch_size: int,
        env: gym.GoalEnv
    ) -> frozen_dict.FrozenDict:
        """Sample a batch of HER samples (online).

        Args:
            batch_size: Number of element to sample (only used for online sampling)
        Returns:
            Samples.
        """
        if hasattr(self.np_random, 'integers'):
            indx = self.np_random.integers(len(self), size=batch_size)
        else:
            indx = self.np_random.randint(len(self), size=batch_size)
        batch = dict()
        for k in self.dataset_dict.keys():
            # DroQ cannot handle dict spaces as of yet.
            if k == "observations" or k == "next_observations":
                batch[k] = np.concatenate((
                    self.dataset_dict[k]["observation"][indx],
                    self.dataset_dict[k]["desired_goal"][indx]),  axis=-1)
            else:
                if isinstance(self.dataset_dict[k], dict):
                    batch[k] = _sample(self.dataset_dict[k], indx)
                else:
                    batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def sample_non_frozen(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None
    ) -> DatasetDict:
        """Sample a non-frozen batch."""
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return batch

    def get_buffer_indices(
        self,
        episode_indices: np.ndarray,
        transition_indices: np.ndarray
    ) -> np.ndarray:
        """Return the buffer index from the given episode + transition.

        Args:
            episode_indices: Episode indices to use.
            transitions_indices: Transition indices to use.
        Returns:
            Buffer indices
        """
        return np.add(self.episode_start_ids[episode_indices], transition_indices)

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        Args:
            episode_indices: Episode indices to use.
            her_indices: HER indices.
            transitions_indices: Transition indices to use.
        Returns:
            sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            assert self.max_episode_stored > 0
            her_transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            if self.max_episode_stored == 0:
                her_transitions_indices = np.random.randint(
                    transitions_indices[her_indices] + 1,
                    np.full([episode_indices.shape[0]], self.current_idx, dtype=np.int32)[her_episode_indices]
                )
            else:
                her_transitions_indices = np.random.randint(
                    transitions_indices[her_indices] + 1,
                    self.episode_lengths[her_episode_indices]
                )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            if self.max_episode_stored == 0:
                her_transitions_indices = np.random.randint(np.full([episode_indices.shape[0]], self.current_idx,
                                                                    dtype=np.int32)[her_episode_indices])
            else:
                her_transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")
        her_buffer_indices = self.get_buffer_indices(episode_indices[her_indices], her_transitions_indices)

        return self.sample_non_frozen(
            her_buffer_indices.shape[0],
            ["observations"],
            her_buffer_indices)["observations"]["achieved_goal"]

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[gym.GoalEnv] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(
        reward: np.ndarray,
        env: Optional[gym.GoalEnv] = None
    ) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
