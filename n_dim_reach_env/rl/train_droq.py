"""This file describes the training functionality for DroQ with HER.

Author: Jakob Thumm
Date: 14.10.2022
"""
import os
import pickle
import shutil
from typing import Optional
import tqdm
import gym  # noqa: F401
import struct
import numpy as np
import wandb

import jax  # noqa: F401
import tensorflow as tf  # noqa: F401

from flax.training import checkpoints

from gym import spaces

from n_dim_reach_env.rl.util.dict_conversion import single_obs
from n_dim_reach_env.rl.util.action_scaling import scale_action, unscale_action

from n_dim_reach_env.rl.agents import SACLearner
from n_dim_reach_env.rl.data import ReplayBuffer
from n_dim_reach_env.rl.data.her_replay_buffer import HEReplayBuffer
from n_dim_reach_env.rl.evaluation import evaluate  # noqa: F401


def train_droq(
    env: gym.Env,
    observation_space: spaces.Space,
    dict_obs: bool,
    seed: int = 0,
    agent_kwargs: dict = {},
    max_ep_len: int = 1000,
    max_steps: int = 1000000,
    start_steps: int = 10000,
    use_her: bool = False,
    n_her_samples: int = 4,
    goal_selection_strategy: str = "future",
    handle_timeout_termination: bool = True,
    utd_ratio: float = 1,
    batch_size: int = 256,
    buffer_size: int = 1000000,
    eval_interval: int = 10000,
    eval_episodes: int = 5,
    eval_callback: Optional[callable] = None,
    load_episode: int = -1,
    run_id: str = "default",
    use_tqdm: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "n-dim-reach",
    wandb_cfg: dict = {},
    wandb_sync_tensorboard: bool = True,
    wandb_monitor_gym: bool = True,
    wandb_save_code: bool = False
):
    """Train a DroQ agent on a given environment.

    Args:
        env (gym.Env): The environment to train on.
        observation_space (spaces.Space): The observation space of the environment.
        dict_obs (bool): Whether the environment uses dict observations.
        seed (int, optional): The seed to use for the environment and the agent. Defaults to 0.
        agent_kwargs (dict, optional): Additional keyword arguments to pass to the agent. Defaults to {}.
        max_ep_len (int, optional): The maximum episode length. Defaults to 1000.
        max_steps (int, optional): The maximum number of steps to train for. Defaults to 1000000.
        start_steps (int, optional): The number of steps to take random actions before training. Defaults to 10000.
        use_her (bool, optional): Whether to use hindsight experience replay. Defaults to False.
        n_her_samples (int, optional): The number of HER samples to generate per transition. Defaults to 4.
        goal_selection_strategy (str, optional): The goal selection strategy to use for HER. Defaults to "future".
        handle_timeout_termination (bool, optional): Whether to handle the timeout termination signal. Defaults to True.
        utd_ratio (float, optional): The update to data ratio. Defaults to 1.
        batch_size (int, optional): The batch size to use for training. Defaults to 256.
        buffer_size (int, optional): The size of the replay buffer. Defaults to 1000000.
        eval_interval (int, optional): The number of steps between evaluations. Defaults to 10000.
        eval_episodes (int, optional): The number of episodes to evaluate for. Defaults to 5.
        eval_callback (Optional[callable], optional): A callback to call after the evaluation runs. Defaults to None.
        load_episode (int, optional): The episode to load the agent from. Defaults to -1.
        run_id (str, optional): The run id to use for wandb. Defaults to "default".
        use_tqdm (bool, optional): Whether to use tqdm for progress bars. Defaults to True.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.
        wandb_project (str, optional): The wandb project to use. Defaults to "n-dim-reach".
        wandb_cfg (dict, optional): Additional wandb config. Defaults to {}.
        wandb_sync_tensorboard (bool, optional): Whether to sync wandb with tensorboard. Defaults to True.
        wandb_monitor_gym (bool, optional): Whether to monitor the gym environment with wandb. Defaults to True.
        wandb_save_code (bool, optional): Whether to save the code to wandb. Defaults to False.
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
    print(jax.devices())
    if load_episode == -1:
        if use_wandb:
            run = wandb.init(
                project=wandb_project,
                config=wandb_cfg,
                sync_tensorboard=wandb_sync_tensorboard,
                monitor_gym=wandb_monitor_gym,
                save_code=wandb_save_code,
            )
        else:
            run = struct
            run.id = int(np.random.rand(1) * 100000)
        agent = SACLearner.create(
            seed=seed,
            observation_space=observation_space,
            action_space=env.action_space,
            **agent_kwargs
        )

        chkpt_dir = 'saved/checkpoints/' + str(run.id)
        os.makedirs(chkpt_dir, exist_ok=True)
        buffer_dir = 'saved/buffers/' + str(run.id)

        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
        start_i = 0
        if use_her:
            assert (isinstance(env.observation_space, spaces.Dict)), "HER requires dict observations"
            replay_buffer = HEReplayBuffer(
                env=env,
                observation_space=env.observation_space,
                action_space=env.action_space,
                capacity=buffer_size,
                achieved_goal_space=env.observation_space["achieved_goal"],
                desired_goal_space=env.observation_space["desired_goal"],
                next_observation_space=env.observation_space,
                max_episode_length=max_ep_len,
                n_sampled_goal=n_her_samples,
                goal_selection_strategy=goal_selection_strategy,
                handle_timeout_termination=handle_timeout_termination)
        else:
            replay_buffer = ReplayBuffer(
                observation_space,
                env.action_space,
                buffer_size
            )
        replay_buffer.seed(seed)
    else:
        if use_wandb:
            run = wandb.init(
                project=wandb_project,
                config=wandb_cfg,
                sync_tensorboard=wandb_sync_tensorboard,
                monitor_gym=wandb_monitor_gym,
                save_code=wandb_save_code,
                resume="must",
                id=run_id,
            )
        else:
            run = struct
            run.id = run_id
        agent = SACLearner.create(
            seed=seed,
            observation_space=observation_space,
            action_space=env.action_space,
            **agent_kwargs
        )
        chkpt_dir = 'saved/checkpoints/' + str(run.id)
        os.makedirs(chkpt_dir, exist_ok=True)
        buffer_dir = 'saved/buffers/' + str(run.id)
        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
        start_i = int(last_checkpoint.split('_')[-1])
        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    observation, done = env.reset(), False

    logging_info = {
        "reward": 0,
        "max_reward": 0,
        "n_collision": 0,
        "n_action_resamples": 0,
        "n_goal_reached": 0,
        "length": 0
    }
    eval_at_next_done = False
    for i in tqdm.tqdm(range(start_i, int(max_steps)),
                       smoothing=0.1,
                       disable=not tqdm):
        if i < start_steps:
            action = env.action_space.sample()
        else:
            if dict_obs:
                action_observation = single_obs(observation)
            else:
                action_observation = observation
            action, agent = agent.sample_actions(action_observation)
            # Agent outputs action in [-1, 1] but we want to step in [low, high]
            action = unscale_action(action,
                                    env.action_space.low,
                                    env.action_space.high)
        next_observation, reward, done, info = env.step(action)
        # Logging
        logging_info["reward"] += reward
        logging_info["max_reward"] = max(logging_info["max_reward"], reward)
        logging_info["n_collision"] = info["n_collision"]
        logging_info["n_action_resamples"] = info["action_resamples"]
        logging_info["n_goal_reached"] = info["n_goal_reached"]
        logging_info["length"] += 1

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        # The action is in [low, high] space, but the agent learns in [-1, 1] space.
        insert_action = scale_action(action,
                                     env.action_space.low,
                                     env.action_space.high)
        # Also any adjusted action must be scaled.
        if "action" in info:
            info["action"] = scale_action(info["action"],
                                          env.action_space.low,
                                          env.action_space.high)
        if use_her:
            replay_buffer.insert(
                data_dict=dict(observations=observation,
                               actions=insert_action,
                               rewards=reward,
                               masks=mask,
                               dones=done,
                               next_observations=next_observation),
                infos=info,
                env=env)
        else:
            if dict_obs:
                replay_buffer.insert(
                    dict(observations=single_obs(observation),
                         actions=insert_action,
                         rewards=reward,
                         masks=mask,
                         dones=done,
                         next_observations=single_obs(next_observation)))
            else:
                replay_buffer.insert(
                    dict(observations=observation,
                         actions=insert_action,
                         rewards=reward,
                         masks=mask,
                         dones=done,
                         next_observations=next_observation))
        observation = next_observation
        if (i+1) % eval_interval == 0:
            eval_at_next_done = True
        if i >= start_steps:
            if use_her:
                batch = replay_buffer.sample(
                        batch_size=batch_size*utd_ratio,
                        env=env
                    )
            else:
                batch = replay_buffer.sample(
                    batch_size*utd_ratio
                )
            agent, update_info = agent.update(batch, utd_ratio)
            if use_wandb and done:  # i % log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)
        if done:
            observation, done = env.reset(), False
            if use_wandb:
                for k, v in logging_info.items():
                    wandb.log({f'training/{k}': v}, step=i)
                wandb.log({
                    'training/avg_reward':
                        logging_info["reward"]/logging_info["length"]
                }, step=i)
            print(logging_info)
            logging_info = {
                "reward": 0,
                "max_reward": 0,
                "n_collision": 0,
                "n_action_resamples": 0,
                "n_goal_reached": 0,
                "length": 0
            }
            if eval_at_next_done:
                env = gym.wrappers.RecordEpisodeStatistics(
                    env,
                    deque_size=eval_episodes
                )
                for _ in range(eval_episodes):
                    observation, done = env.reset(), False
                    while not done:
                        if dict_obs:
                            action_observation = single_obs(observation)
                        else:
                            action_observation = observation
                        action = agent.eval_actions(action_observation)
                        action = unscale_action(action,
                                                env.action_space.low,
                                                env.action_space.high)
                        observation, _, done, _ = env.step(action)
                eval_info = {
                    'return': np.mean(env.return_queue),
                    'length': np.mean(env.length_queue)
                }
                if eval_callback is not None:
                    eval_callback(eval_info["return"])
                if use_wandb:
                    for k, v in eval_info.items():
                        wandb.log({f'evaluation/{k}': v}, step=i)
                eval_at_next_done = False
                observation, done = env.reset(), False
                checkpoints.save_checkpoint(chkpt_dir,
                                            agent,
                                            step=i + 1,
                                            keep=20,
                                            overwrite=True)
                try:
                    shutil.rmtree(buffer_dir)
                except Exception:
                    pass

                os.makedirs(buffer_dir, exist_ok=True)
                with open(
                    os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb'
                ) as f:
                    pickle.dump(replay_buffer, f)
    if use_wandb:
        run.finish()
    env.close()
