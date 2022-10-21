#!/usr/bin/env python
"""This file describes the plotting of the Q function of DroQ.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    15.10.22 JT Created File.
"""
import os
from copy import copy
from tkinter import N
from typing import Any, Dict
import gym  # noqa: F401
import numpy as np
import jax
import pickle
import matplotlib.pyplot as plt

from flax.training import checkpoints

import hydra
from hydra.core.config_store import ConfigStore

from gym.wrappers import TimeLimit
from gym import spaces

from n_dim_reach_env.envs.reach_env import ReachEnv
from n_dim_reach_env.rl.optimization.optimize_hyperparameters import optimize_hyperparameters  # noqa: F401
from n_dim_reach_env.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper  # noqa: E501
from n_dim_reach_env.conf.config_droq import DroQTrainingConfig, EnvConfig

from n_dim_reach_env.train.train_reach_droq import create_env, get_observation_space, has_dict_obs

from n_dim_reach_env.rl.agents import SACLearner
from n_dim_reach_env.rl.data import ReplayBuffer
from n_dim_reach_env.rl.data.her_replay_buffer import HEReplayBuffer

cs = ConfigStore.instance()
cs.store(name="droq_config", node=DroQTrainingConfig)


@hydra.main(config_path="../conf", config_name="conf_droq")
def main(cfg: DroQTrainingConfig):
    """Plot the Q function of the DroQ Agent."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
    print(jax.devices())

    # << Training >>
    env = create_env(cfg.env)
    observation_space = get_observation_space(env)
    has_dict = has_dict_obs(env)
    action_space = env.action_space
    run_id = cfg.train.run_id
    agent_kwargs = {
        "actor_lr": cfg.droq.actor_lr,
        "critic_lr": cfg.droq.critic_lr,
        "temp_lr": cfg.droq.temp_lr,
        "hidden_dims": cfg.droq.hidden_dims,
        "discount": cfg.droq.discount,
        "tau": cfg.droq.tau,
        "num_qs": cfg.droq.num_qs,
        "num_min_qs": cfg.droq.num_min_qs,
        "critic_dropout_rate": cfg.droq.critic_dropout_rate,
        "critic_layer_norm": cfg.droq.critic_layer_norm,
        "target_entropy": cfg.droq.target_entropy,
        "init_temperature": cfg.droq.init_temperature,
        "sampled_backup": cfg.droq.sampled_backup
    }
    agent = SACLearner.create(
        cfg.env.seed,
        observation_space,
        action_space,
        **agent_kwargs)
    # chkpt_dir = 'saved/checkpoints/' + str(run_id)
    # buffer_dir = 'saved/buffers/' + str(run_id)
    chkpt_dir = '/home/jakob/Promotion/code/n-dim-reach-env/outputs/2022-10-14/18-23-54/saved/checkpoints/30706'
    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
    # start_i = int(last_checkpoint.split('_')[-1])
    agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
    # Test actor
    n_dim = cfg.env.n_dim
    observation = observation_space.sample()
    if n_dim == 1:
        print("Dimension 1 not supported.")
        return
    elif n_dim > 2:
        # Set all other dimensions to "goal reached"
        observation[2:n_dim] = observation[n_dim+2:2*n_dim]
    action = agent.eval_actions(observation)
    # Test critic
    n_points = 10
    n_actions = 8
    q_values = np.zeros([n_points, n_points])
    best_actions = np.zeros([n_points, n_points, 2])
    positions = np.linspace(-1 + 1/n_points, 1-1/n_points, n_points)
    phis = np.pi * np.linspace(-1 + 1/n_actions, 1 - 1/n_actions, n_actions)
    actions = np.zeros([n_actions, 2])
    actions[:, 0] = np.cos(phis)
    actions[:, 1] = np.sin(phis)
    zero_action = np.zeros(action.shape[0])
    a = np.repeat(zero_action[np.newaxis,:], n_actions, axis = 0)
    a[:, 0:2] = actions
    for i in range(n_points):
        for j in range(n_points):
            obs = observation.copy()
            obs[0:2] = [positions[i], positions[j]]
            o = np.repeat(obs[np.newaxis,:], n_actions, axis = 0)
            key, rng = jax.random.split(agent.rng)
            key2, rng = jax.random.split(rng)
            q_vals = np.mean(agent.critic.apply_fn(
                {'params': agent.critic.params},
                o,
                a,
                True,
                rngs={'dropout': key2})._value, axis=0)
            best = np.argmax(q_vals)
            best_actions[i, j] = actions[best]
            q_values[i, j] = q_vals[best]

    # Plot a heatmap of the Q function
    plt.imshow(q_values, cmap='hot', interpolation='nearest')
    for i in range(n_points):
        for j in range(n_points):
            plt.arrow(i, j, best_actions[i, j, 0], best_actions[i, j, 1], width=0.01)
    plt.plot(observation[n_dim], observation[n_dim+1], 'bo')
    plt.show()
    stop=0


if __name__ == "__main__":
    main()
