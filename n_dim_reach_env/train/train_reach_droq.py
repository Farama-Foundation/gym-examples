#!/usr/bin/env python
"""This file describes the training functionality for DroQ with HER.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    5.9.22 JT Created File.
"""
from copy import copy
import gym  # noqa: F401
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore

from gym.wrappers import TimeLimit
from gym import spaces

from n_dim_reach_env.envs.reach_env import ReachEnv  # noqa: F401
from n_dim_reach_env.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper  # noqa: E501
from n_dim_reach_env.conf.config_droq import DroQTrainingConfig

from n_dim_reach_env.rl.train_droq import train_droq

cs = ConfigStore.instance()
cs.store(name="droq_config", node=DroQTrainingConfig)


@hydra.main(config_path="../conf", config_name="conf_droq")
def main(cfg: DroQTrainingConfig):
    """Train a SAC agent on the n dimensional reach environment."""
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
        observation_space = copy(env.observation_space)
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
        observation_space = spaces.Box(low=lows, high=highs)
    else:
        observation_space = env.observation_space
        dict_obs = False
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
    train_droq(
        env=env,
        observation_space=observation_space,
        dict_obs=dict_obs,
        seed=cfg.env.seed,
        agent_kwargs=agent_kwargs,
        max_ep_len=cfg.env.max_ep_len,
        max_steps=cfg.train.max_steps,
        start_steps=cfg.droq.start_steps,
        use_her=cfg.droq.use_her,
        n_her_samples=cfg.droq.n_her_samples,
        goal_selection_strategy=cfg.droq.goal_selection_strategy,
        handle_timeout_termination=cfg.droq.handle_timeout_termination,
        utd_ratio=cfg.droq.utd_ratio,
        batch_size=cfg.droq.batch_size,
        buffer_size=cfg.droq.buffer_size,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
        load_episode=cfg.train.load_episode,
        run_id=cfg.train.run_id,
        use_tqdm=cfg.train.tqdm,
        use_wandb=cfg.train.use_wandb,
        wandb_project=cfg.train.wandb_project,
        wandb_cfg=cfg,
        wandb_sync_tensorboard=True,
        wandb_monitor_gym=True,
        wandb_save_code=False,
    )


if __name__ == "__main__":
    main()