"""This file fixes an issue in SB3 implementation of HER.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    2.5.22 JT Formatted docstrings
"""
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from gym.spaces import Box

from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize  # noqa: F401
from stable_baselines3.her.goal_selection_strategy import (  # noqa: F401
    KEY_TO_GOAL_STRATEGY,  # noqa: F401
    GoalSelectionStrategy,  # noqa: F401
)  # noqa: F401
from n_dim_reach_env.utils.tictoc import tic, toc

def custom_add(
    self,
    obs: Dict[str, np.ndarray],
    next_obs: Dict[str, np.ndarray],
    action: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    infos: List[Dict[str, Any]],
) -> None:
    """Add new transitions (s, a, s', r) to the replay buffer.

    Checks if the key "action" is in the info dict and
    replaces the action in the transition with the action of the info dict.

    Args:
        obs: observation dict (s).
        next_obs: next observation dict (s').
        action: action commanded by the agent (not necessarly the executed action!) (a).
        reward: the received reward (r).
        done: if the episode was done after the transition.
        infos: info dictionary (may contain the truely executed action).
    """
    if self.current_idx == 0 and self.full:
        # Clear info buffer
        self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

    # Remove termination signals due to timeout
    if self.handle_timeout_termination:
        done_ = done * (
            1 - np.array([info.get("TimeLimit.truncated", False) for info in infos])
        )
    else:
        done_ = done

    self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
    self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
    self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
    # If the environment changed the action.
    if "action" in infos[0]:
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, Box):
            # THIS ASSUMES TANH ACTIVATION FUNCTION FOR THE POLICY NETWORK!!!
            scaled_action = (
                2.0
                * (
                    (infos[0]["action"] - self.action_space.low)
                    / (self.action_space.high - self.action_space.low)
                )
                - 1.0
            )
            action = np.clip(scaled_action, -1, 1)
        else:
            action = infos["action"]
    self._buffer["action"][self.pos][self.current_idx] = action
    self._buffer["done"][self.pos][self.current_idx] = done_
    self._buffer["reward"][self.pos][self.current_idx] = reward
    self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
    self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs[
        "achieved_goal"
    ]
    self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs[
        "desired_goal"
    ]

    # When doing offline sampling
    # Add real transition to normal replay buffer
    if self.replay_buffer is not None:
        self.replay_buffer.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )

    self.info_buffer[self.pos].append(infos)

    # update current pointer
    self.current_idx += 1

    self.episode_steps += 1

    if done or self.episode_steps >= self.max_episode_length:
        self.store_episode()
        if not self.online_sampling:
            # sample virtual transitions and store them in replay buffer
            self._sample_her_transitions()
            # clear storage for current episode
            self.reset()

        self.episode_steps = 0


def _custom_sample_transitions(
    self,
    batch_size: Optional[int],
    maybe_vec_env: Optional[VecNormalize],
    online_sampling: bool,
    n_sampled_goal: Optional[int] = None,
) -> Union[
    DictReplayBufferSamples,
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray],
]:
    """Sample a set of transitions from the replay buffer with HER strategy future.

    This monkey patch correctly updates the done flag of the HER transitions.

    Args:
        batch_size: Number of element to sample (only used for online sampling)
        env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        online_sampling: Using online_sampling for HER or not.
        n_sampled_goal: Number of sampled goals for replay. (offline sampling)
    Returns
        Samples.
    """
    if not hasattr(self,'reward_fn'):
        target_env = self.env._get_target_envs(0)[0]
        self.reward_fn = getattr(target_env, "compute_reward")
        self.done_fn = getattr(target_env, "compute_done")
    # Select which episodes to use
    if online_sampling:
        assert (
            batch_size is not None
        ), "No batch_size specified for online sampling of HER transitions"
        # Do not sample the episode with index `self.pos` as the episode is invalid
        if self.full:
            episode_indices = (
                np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
            ) % self.n_episodes_stored
        else:
            episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
        # A subset of the transitions will be relabeled using HER algorithm
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
    else:
        assert (
            maybe_vec_env is None
        ), "Transitions must be stored unnormalized in the replay buffer"
        assert (
            n_sampled_goal is not None
        ), "No n_sampled_goal specified for offline sampling of HER transitions"
        # Offline sampling: there is only one episode stored
        episode_length = self.episode_lengths[0]
        # we sample n_sampled_goal per timestep in the episode (only one is stored).
        episode_indices = np.tile(0, (episode_length * n_sampled_goal))
        # we only sample virtual transitions
        # as real transitions are already stored in the replay buffer
        her_indices = np.arange(len(episode_indices))

    ep_lengths = self.episode_lengths[episode_indices]

    # Special case when using the "future" goal sampling strategy
    # we cannot sample all transitions, we have to remove the last timestep
    if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
        # restrict the sampling domain when ep_lengths > 1
        # otherwise filter out the indices
        her_indices = her_indices[ep_lengths[her_indices] > 1]
        ep_lengths[her_indices] -= 1

    if online_sampling:
        # Select which transitions to use
        transitions_indices = np.random.randint(ep_lengths)
    else:
        if her_indices.size == 0:
            # Episode of one timestep, not enough for using the "future" strategy
            # no virtual transitions are created in that case
            return {}, {}, np.zeros(0), np.zeros(0)
        else:
            # Repeat every transition index n_sampled_goals times
            # to sample n_sampled_goal per timestep in the episode (only one is stored).
            # Now with the corrected episode length when using "future" strategy
            transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
            episode_indices = episode_indices[transitions_indices]
            her_indices = np.arange(len(episode_indices))

    # get selected transitions
    transitions = {
        key: self._buffer[key][episode_indices, transitions_indices].copy()
        for key in self._buffer.keys()
    }

    # sample new desired goals and relabel the transitions
    new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
    transitions["desired_goal"][her_indices] = new_goals

    # Convert info buffer to numpy array
    transitions["info"] = np.array(
        [
            self.info_buffer[episode_idx][transition_idx]
            for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
        ]
    )
    # Edge case: episode of one timesteps with the future strategy
    # no virtual transition can be created
    if len(her_indices) > 0:
        # Vectorized computation of the new reward
        her_next_achieved_goal = transitions["next_achieved_goal"][her_indices, 0]
        her_desired_goal = transitions["desired_goal"][her_indices, 0]
        her_info = transitions["info"][her_indices, 0]
        re = self.reward_fn(her_next_achieved_goal, her_desired_goal, her_info)
        transitions["reward"][her_indices, 0] = re
        # Vectorized computation of the new done values
        transitions["done"][her_indices, 0] = self.done_fn(
            her_next_achieved_goal,
            her_desired_goal,
            her_info
        )
    # concatenate observation with (desired) goal
    observations = self._normalize_obs(transitions, maybe_vec_env)

    # HACK to make normalize obs and `add()` work with the next observation
    next_observations = {
        "observation": transitions["next_obs"],
        "achieved_goal": transitions["next_achieved_goal"],
        # The desired goal for the next observation must be the same as the previous one
        "desired_goal": transitions["desired_goal"],
    }
    next_observations = self._normalize_obs(next_observations, maybe_vec_env)
    if online_sampling:
        next_obs = {
            key: self.to_torch(next_observations[key][:, 0, :])
            for key in self._observation_keys
        }

        normalized_obs = {
            key: self.to_torch(observations[key][:, 0, :])
            for key in self._observation_keys
        }
        samples = DictReplayBufferSamples(
            observations=normalized_obs,
            actions=self.to_torch(transitions["action"]),
            next_observations=next_obs,
            dones=self.to_torch(transitions["done"]),
            rewards=self.to_torch(
                self._normalize_reward(transitions["reward"], maybe_vec_env)
            ))
        return samples
    else:
        return (
            observations,
            next_observations,
            transitions["action"],
            transitions["reward"],
        )
