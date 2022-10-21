#!/usr/bin/env python
"""This file describes the plotting of the Q function of DroQ.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    15.10.22 JT Created File.
"""
import os
import gym  # noqa: F401
import numpy as np
import jax
# import pickle
import matplotlib.pyplot as plt

from flax.training import checkpoints

import hydra
from hydra.core.config_store import ConfigStore

from n_dim_reach_env.envs.reach_env import ReachEnv  # noqa: F401
from n_dim_reach_env.rl.optimization.optimize_hyperparameters import optimize_hyperparameters  # noqa: F401
from n_dim_reach_env.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper  # noqa: F401
from n_dim_reach_env.conf.config_droq import DroQTrainingConfig

from n_dim_reach_env.train.train_reach_droq import create_env, get_observation_space
from n_dim_reach_env.rl.util.action_scaling import unscale_action

from n_dim_reach_env.rl.agents import SACLearner
# from n_dim_reach_env.rl.data import ReplayBuffer
# from n_dim_reach_env.rl.data.her_replay_buffer import HEReplayBuffer

cs = ConfigStore.instance()
cs.store(name="droq_config", node=DroQTrainingConfig)

PLOT_DIMENSIONS = [0, 3]
N_PLOTS = 5
N_POINTS = 10
N_ACTIONS = 8


@hydra.main(config_path="../conf", config_name="conf_droq")
def main(cfg: DroQTrainingConfig):
    """Plot the Q function of the DroQ Agent."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
    print(jax.devices())

    # << Training >>
    env = create_env(cfg.env)
    observation_space = get_observation_space(env)
    # has_dict = has_dict_obs(env)
    action_space = env.action_space
    assert cfg.train.load_from_folder is not None
    chkpt_dir = cfg.train.load_from_folder + 'saved/checkpoints/'
    buffer_dir = cfg.train.load_from_folder + 'saved/buffers/'
    d = os.listdir(chkpt_dir)[0]
    assert os.path.isdir(chkpt_dir + d)
    chkpt_dir = chkpt_dir + d
    buffer_dir = buffer_dir + d

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
    if cfg.train.load_checkpoint == -1:
        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
    else:
        last_checkpoint = chkpt_dir + '/checkpoint_' + str(cfg.train.load_checkpoint)
    # start_i = int(last_checkpoint.split('_')[-1])
    agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
    # with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
    #     replay_buffer = pickle.load(f)
    # Test actor
    n_dim = cfg.env.n_dim
    observation = observation_space.sample()
    assert n_dim >= 2, "n_dim must be at least 2"
    if n_dim > 2:
        # Set all other dimensions to "goal reached"
        for i in range(n_dim):
            if i not in PLOT_DIMENSIONS:
                observation[i] = observation[i + n_dim]
    action = agent.eval_actions(observation)
    # Test critic
    q_values = np.zeros([N_POINTS, N_POINTS])
    best_actions = np.zeros([N_POINTS, N_POINTS, 2])
    positions = np.linspace(-1 + 1/N_POINTS, 1-1/N_POINTS, N_POINTS)
    phis = np.pi * np.linspace(-1 + 1/N_ACTIONS, 1 - 1/N_ACTIONS, N_ACTIONS)
    actions = np.zeros([N_ACTIONS, 2])
    actions[:, 0] = np.cos(phis)
    actions[:, 1] = np.sin(phis)
    zero_action = np.zeros(action.shape[0])
    a = np.repeat(zero_action[np.newaxis, :], N_ACTIONS, axis=0)
    a[:, PLOT_DIMENSIONS] = actions
    for i in range(N_POINTS):
        for j in range(N_POINTS):
            obs = observation.copy()
            obs[PLOT_DIMENSIONS] = [positions[i], positions[j]]
            o = np.repeat(obs[np.newaxis, :], N_ACTIONS, axis=0)
            key, rng = jax.random.split(agent.rng)
            key2, rng = jax.random.split(rng)
            q_vals = np.min(agent.critic.apply_fn(
                {'params': agent.critic.params},
                o,
                a,
                True,
                rngs={'dropout': key2})._value, axis=0)
            best = np.argmax(q_vals)
            best_actions[i, j] = unscale_action(
                actions[best],
                low=action_space.low[PLOT_DIMENSIONS],
                high=action_space.high[PLOT_DIMENSIONS])
            q_values[i, j] = q_vals[best]

    # Plot a heatmap of the Q function
    # left, right, bottom, top
    extent = [-1, 1, 1, -1]
    plt.imshow(q_values, cmap='hot', interpolation='nearest', extent=extent)
    for i in range(N_POINTS):
        for j in range(N_POINTS):
            plt.arrow(positions[i], positions[j], best_actions[i, j, 0], best_actions[i, j, 1], width=0.01)
    plt.plot(observation[PLOT_DIMENSIONS[0]+n_dim], observation[PLOT_DIMENSIONS[1]+n_dim], 'bo')
    plt.colorbar()
    plt.show()
    stop=0


if __name__ == "__main__":
    main()
