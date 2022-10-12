#!/usr/bin/env python
"""This file describes the training functionality for DroQ with HER.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    5.9.22 JT Created File.
"""
from copy import copy
import os
import pickle
import shutil
import tqdm
import gym  # noqa: F401
import struct
import numpy as np
import wandb

import hydra
from hydra.core.config_store import ConfigStore

import jax  # noqa: F401
import tensorflow as tf  # noqa: F401

from flax.training import checkpoints

from gym.wrappers import TimeLimit
from gym import spaces

from n_dim_reach_env.envs.reach_env import ReachEnv  # noqa: F401
from n_dim_reach_env.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper  # noqa: E501
from n_dim_reach_env.conf.config_droq import DroQTrainingConfig
from n_dim_reach_env.wrappers.HER_buffer_add_monkey_patch import single_obs

from n_dim_reach_env.rl.agents import SACLearner
from n_dim_reach_env.rl.data import ReplayBuffer
from n_dim_reach_env.rl.data.her_replay_buffer import HEReplayBuffer
from n_dim_reach_env.rl.evaluation import evaluate  # noqa: F401

cs = ConfigStore.instance()
cs.store(name="droq_config", node=DroQTrainingConfig)


@hydra.main(config_path="../conf", config_name="conf_droq")
def main(cfg: DroQTrainingConfig):
    """Train a SAC agent on the n dimensional reach environment."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
    print(jax.devices())
    print(cfg)
    env = gym.make(cfg.env.id,
                   n_dim=cfg.env.n_dim,
                   max_action=cfg.env.max_action,
                   goal_distance=cfg.env.goal_distance,
                   done_on_collision=cfg.env.done_on_collision,
                   randomize_env=cfg.env.randomize_env,
                   collision_reward=cfg.env.collision_reward,
                   goal_reward=cfg.env.goal_reward,
                   step_reward=cfg.env.step_reward,
                   reward_shaping=cfg.env.reward_shaping,
                   render_mode=cfg.env.render_mode,
                   seed=cfg.env.seed)
    env = TimeLimit(env, cfg.env.max_ep_len)
    env = CollisionPreventionWrapper(env,
                                     replace_type=cfg.env.replace_type,
                                     n_resamples=cfg.env.n_resamples,
                                     punishment=cfg.env.punishment)
    # If the observation space is of type dict,
    # change the observation space. DroQ cannot handle dicts right now.
    if isinstance(env.observation_space, spaces.Dict):
        dict_observation_space = copy(env.observation_space)
        dict_obs = True
        lows = None
        highs = None
        for key in env.observation_space.spaces:
            if key != "achieved_goal":
                if lows is None:
                    lows = np.array(env.observation_space.spaces[key].low)
                    highs = np.array(env.observation_space.spaces[key].high)
                else:
                    lows = np.append(lows, env.observation_space.spaces[key].low)
                    highs = np.append(highs, env.observation_space.spaces[key].high)
        env.observation_space = spaces.Box(low=lows, high=highs)

    # << Training >>
    if cfg.train.load_episode == -1:
        if cfg.train.use_wandb:
            run = wandb.init(
                project="n-dim-reach",
                config=cfg,
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=False,
            )
        else:
            run = struct
            run.id = int(np.random.rand(1) * 100000)
        agent = SACLearner.create(
            seed=cfg.env.seed,
            observation_space=env.observation_space,
            action_space=env.action_space,
            actor_lr=cfg.droq.actor_lr,
            critic_lr=cfg.droq.critic_lr,
            temp_lr=cfg.droq.temp_lr,
            hidden_dims=cfg.droq.hidden_dims,
            discount=cfg.droq.discount,
            tau=cfg.droq.tau,
            num_qs=cfg.droq.num_qs,
            num_min_qs=cfg.droq.num_min_qs,
            critic_dropout_rate=cfg.droq.critic_dropout_rate,
            critic_layer_norm=cfg.droq.critic_layer_norm,
            target_entropy=cfg.droq.target_entropy,
            init_temperature=cfg.droq.init_temperature,
            sampled_backup=cfg.droq.sampled_backup
        )

        chkpt_dir = 'saved/checkpoints/' + str(run.id)
        os.makedirs(chkpt_dir, exist_ok=True)
        buffer_dir = 'saved/buffers/' + str(run.id)

        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
        start_i = 0
        if cfg.droq.use_HER:
            replay_buffer = HEReplayBuffer(
                env=env,
                observation_space=dict_observation_space,
                action_space=env.action_space,
                capacity=cfg.droq.buffer_size,
                achieved_goal_space=dict_observation_space["achieved_goal"],
                desired_goal_space=dict_observation_space["desired_goal"],
                next_observation_space=dict_observation_space,
                max_episode_length=cfg.env.max_ep_len,
                n_sampled_goal=cfg.droq.n_her_samples,
                goal_selection_strategy=cfg.droq.goal_selection_strategy,
                handle_timeout_termination=cfg.droq.handle_timeout_termination)
        else:
            replay_buffer = ReplayBuffer(
                env.observation_space,
                env.action_space,
                cfg.droq.buffer_size
            )
        replay_buffer.seed(cfg.env.seed)
    else:
        if cfg.train.use_wandb:
            run = wandb.init(
                project="human_comfort_learning",
                config=cfg,
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=False,
                resume="must",
                id=cfg.train.run_id,
            )
        else:
            run = struct
            run.id = cfg.train.run_id
        agent = SACLearner.create(
            seed=cfg.env.seed,
            observation_space=env.observation_space,
            action_space=env.action_space,
            actor_lr=cfg.droq.actor_lr,
            critic_lr=cfg.droq.critic_lr,
            temp_lr=cfg.droq.temp_lr,
            hidden_dims=cfg.droq.hidden_dims,
            discount=cfg.droq.discount,
            tau=cfg.droq.tau,
            num_qs=cfg.droq.num_qs,
            num_min_qs=cfg.droq.num_min_qs,
            critic_dropout_rate=cfg.droq.critic_dropout_rate,
            critic_layer_norm=cfg.droq.critic_layer_norm,
            target_entropy=cfg.droq.target_entropy,
            init_temperature=cfg.droq.init_temperature,
            sampled_backup=cfg.droq.sampled_backup
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
    for i in tqdm.tqdm(range(start_i, int(cfg.train.max_steps)),
                       smoothing=0.1,
                       disable=not cfg.train.tqdm):
        if i < cfg.droq.start_steps:
            action = env.action_space.sample()
        else:
            if dict_obs:
                action_observation = single_obs(observation)
            else:
                action_observation = observation
            action, agent = agent.sample_actions(action_observation)
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
        if cfg.droq.use_HER:
            replay_buffer.insert(
                data_dict=dict(observations=observation,
                               actions=action,
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
                         actions=action,
                         rewards=reward,
                         masks=mask,
                         dones=done,
                         next_observations=single_obs(next_observation)))
            else:
                replay_buffer.insert(
                    dict(observations=observation,
                         actions=action,
                         rewards=reward,
                         masks=mask,
                         dones=done,
                         next_observations=next_observation))
        observation = next_observation
        if (i+1) % cfg.train.eval_interval == 0:
            eval_at_next_done = True
        if i >= cfg.droq.start_steps:
            if cfg.droq.use_HER:
                batch = replay_buffer.sample(
                        batch_size=cfg.droq.batch_size*cfg.droq.utd_ratio,
                        env=env
                    )
            else:
                batch = replay_buffer.sample(
                    cfg.droq.batch_size*cfg.droq.utd_ratio
                )
            agent, update_info = agent.update(batch, cfg.droq.utd_ratio)
            if cfg.train.use_wandb and done:  # i % cfg.train.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)
        if done:
            observation, done = env.reset(), False
            if cfg.train.use_wandb:
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
                    deque_size=cfg.train.eval_episodes
                )
                for _ in range(cfg.train.eval_episodes):
                    observation, done = env.reset(), False
                    while not done:
                        if dict_obs:
                            action_observation = single_obs(observation)
                        else:
                            action_observation = observation
                        action = agent.eval_actions(action_observation)
                        observation, _, done, _ = env.step(action)
                eval_info = {
                    'return': np.mean(env.return_queue),
                    'length': np.mean(env.length_queue)
                }
                if cfg.train.use_wandb:
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
    if cfg.train.use_wandb:
        run.finish()
    env.close()


if __name__ == "__main__":
    main()