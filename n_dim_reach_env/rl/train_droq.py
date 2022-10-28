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
from n_dim_reach_env.rl.data.single_demo_booster import SingleDemoBooster
from n_dim_reach_env.rl.evaluation import evaluate  # noqa: F401


def train_droq(
    env: gym.Env,
    eval_env: gym.Env,
    observation_space: spaces.Space,
    dict_obs: bool,
    seed: int = 0,
    agent_kwargs: dict = {},
    max_ep_len: int = 1000,
    max_steps: int = 1000000,
    start_steps: int = 10000,
    squash_output: bool = True,
    use_her: bool = False,
    n_her_samples: int = 4,
    goal_selection_strategy: str = "future",
    handle_timeout_termination: bool = True,
    boost_single_demo: bool = False,
    boost_single_demo_kwargs: dict = {},
    pre_play_steps: int = 0,
    pre_play_rate: float = 0.0,
    utd_ratio: float = 1,
    batch_size: int = 256,
    buffer_size: int = 1000000,
    eval_interval: int = 10000,
    eval_episodes: int = 5,
    eval_callback: Optional[callable] = None,
    load_checkpoint: int = -1,
    load_from_folder: str = None,
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
        eval_env (gym.Env): The environment to evaluate on.
        observation_space (spaces.Space): The observation space of the environment.
        dict_obs (bool): Whether the environment uses dict observations.
        seed (int, optional): The seed to use for the environment and the agent. Defaults to 0.
        agent_kwargs (dict, optional): Additional keyword arguments to pass to the agent. Defaults to {}.
        max_ep_len (int, optional): The maximum episode length. Defaults to 1000.
        max_steps (int, optional): The maximum number of steps to train for. Defaults to 1000000.
        start_steps (int, optional): The number of steps to take random actions before training. Defaults to 10000.
        squash_output (bool, optional): Whether to squash the output of the actor to [low, high]. Defaults to True.
        use_her (bool, optional): Whether to use hindsight experience replay. Defaults to False.
        n_her_samples (int, optional): The number of HER samples to generate per transition. Defaults to 4.
        goal_selection_strategy (str, optional): The goal selection strategy to use for HER. Defaults to "future".
        handle_timeout_termination (bool, optional): Whether to handle the timeout termination signal. Defaults to True.
        boost_single_demo (bool, optional): Whether to use a single demonstration to boost training. Defaults to False.
        boost_single_demo_kwargs (dict, optional): Additional keyword arguments to pass to the single demonstration
            booster. Defaults to {}.
        utd_ratio (float, optional): The update to data ratio. Defaults to 1.
        batch_size (int, optional): The batch size to use for training. Defaults to 256.
        buffer_size (int, optional): The size of the replay buffer. Defaults to 1000000.
        eval_interval (int, optional): The number of steps between evaluations. Defaults to 10000.
        eval_episodes (int, optional): The number of episodes to evaluate for. Defaults to 5.
        eval_callback (Optional[callable], optional): A callback to call after the evaluation runs. Defaults to None.
        load_checkpoint (int, optional): The checkpoint to load. Defaults to -1.
            Set to -1 to load the latest checkpoint.
        load_from_folder (str, optional): The folder to load the checkpoint from. Defaults to None.
            Set to None to not load from folder.
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
    if pre_play_steps > 0 and not boost_single_demo:
        print("WARNING: Pre-play steps can only be used together with boosting a single demonstration.\
              Disabling pre-play.")
        pre_play_steps = 0

    if load_from_folder is None:
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
        # Single demonstration booster
        if boost_single_demo:
            replay_buffer = SingleDemoBooster(
                env=env,
                replay_buffer=replay_buffer,
                observation_space=observation_space,
                action_space=env.action_space,
                single_demo=None,
                **boost_single_demo_kwargs
            )
    else:
        chkpt_dir = load_from_folder + 'saved/checkpoints/'
        buffer_dir = load_from_folder + 'saved/buffers/'
        d = os.listdir(chkpt_dir)[0]
        if os.path.isdir(chkpt_dir + d):
            run_id = d
            chkpt_dir = chkpt_dir + d
            buffer_dir = buffer_dir + d

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
        if load_checkpoint == -1:
            last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
        else:
            last_checkpoint = chkpt_dir + '/checkpoint_' + str(load_checkpoint)
        start_i = int(last_checkpoint.split('_')[-1])
        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    eval_env = gym.wrappers.RecordEpisodeStatistics(
        eval_env,
        deque_size=eval_episodes
    )
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
    max_pre_play_actions = 0
    pre_play_action_counter = 0
    for i in tqdm.tqdm(range(start_i, int(max_steps)),
                       smoothing=0.1,
                       disable=not tqdm):
        # Start steps with random actions
        if i < start_steps:
            action = env.action_space.sample()
        # Potential pre-play steps
        elif i < start_steps + pre_play_steps and pre_play_action_counter <= max_pre_play_actions:
            action = replay_buffer.get_artificial_action(pre_play_action_counter, observation["achieved_goal"])
            pre_play_action_counter += 1
        # Normal agent actions
        else:
            if dict_obs:
                action_observation = single_obs(observation)
            else:
                action_observation = observation
            action, agent = agent.sample_actions(action_observation)
            # Agent outputs action in [-1, 1] but we want to step in [low, high]
            action = unscale_action(action,
                                    env.action_space.low,
                                    env.action_space.high,
                                    squash_output)
        next_observation, reward, done, info = env.step(action)
        # Logging
        logging_info["reward"] += reward
        logging_info["max_reward"] = max(logging_info["max_reward"], reward)
        logging_info["n_collision"] = info["n_collision"]
        if "action_resample" in info:
            logging_info["n_action_resamples"] = info["action_resamples"]
        logging_info["n_goal_reached"] = info["n_goal_reached"]
        logging_info["length"] += 1

        if not done:
            mask = 1.0
        else:
            if 'TimeLimit.truncated' in info:
                mask = 1.0 if info['TimeLimit.truncated'] else 0.0
            mask = 0.0
        # The action is in [low, high] space, but the agent learns in [-1, 1] space.
        insert_action = scale_action(action,
                                     env.action_space.low,
                                     env.action_space.high,
                                     squash_output)
        # Also any adjusted action must be scaled.
        if "action" in info:
            info["action"] = scale_action(info["action"],
                                          env.action_space.low,
                                          env.action_space.high,
                                          squash_output)
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
                for _ in range(eval_episodes):
                    observation, done = eval_env.reset(), False
                    while not done:
                        if dict_obs:
                            action_observation = single_obs(observation)
                        else:
                            action_observation = observation
                        action = agent.eval_actions(action_observation)
                        action = unscale_action(action,
                                                eval_env.action_space.low,
                                                eval_env.action_space.high,
                                                squash_output)
                        observation, _, done, _ = eval_env.step(action)
                eval_info = {
                    'return': np.mean(eval_env.return_queue),
                    'length': np.mean(eval_env.length_queue)
                }
                print("Eval info:", eval_info)
                if eval_callback is not None:
                    eval_callback(eval_info["return"])
                if use_wandb:
                    for k, v in eval_info.items():
                        wandb.log({f'evaluation/{k}': v}, step=i)
                eval_at_next_done = False
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
            observation, done = env.reset(), False
            if i >= start_steps and i < start_steps + pre_play_steps:
                r = np.random.rand()
                if r < pre_play_rate:
                    n_pre_play_actions = replay_buffer.initialize_artificial_trajectory(
                        p_gen_start=observation["achieved_goal"],
                        p_gen_goal=observation["desired_goal"]
                    )
                    max_pre_play_actions = np.random.randint(low=1, high=n_pre_play_actions)
                    pre_play_action_counter = 0

    if use_wandb:
        run.finish()
    env.close()
