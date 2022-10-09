"""Training function for SB3 SAC on dense rewards."""
import hydra
from hydra.core.config_store import ConfigStore
import gym
from gym.wrappers import TimeLimit
from stable_baselines3.sac.sac import SAC

from n_dim_reach_env.envs.reach_env import ReachEnv  # noqa: F401
from n_dim_reach_env.conf.config import SACTrainingConfig

cs = ConfigStore.instance()
cs.store(name="sac_config", node=SACTrainingConfig)


@hydra.main(config_path="../conf", config_name="conf")
def main(cfg: SACTrainingConfig):
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
    env = TimeLimit(env, cfg.sb3.max_ep_len)
    model = SAC("MultiInputPolicy",
                env,
                buffer_size=cfg.sb3.replay_size,
                verbose=cfg.verbose,
                learning_rate=cfg.sb3.lr,
                learning_starts=cfg.sb3.start_steps,
                batch_size=cfg.sb3.batch_size,
                tau=cfg.sb3.tau,
                gamma=cfg.sb3.gamma,
                train_freq=cfg.sb3.update_every,
                gradient_steps=cfg.sb3.gradient_steps,
                action_noise=cfg.sb3.action_noise,
                optimize_memory_usage=False,
                ent_coef=cfg.sb3.ent_coef,
                target_update_interval=cfg.sb3.target_update_interval,
                target_entropy=cfg.sb3.target_entropy,
                use_sde=cfg.sb3.use_sde,
                sde_sample_freq=cfg.sb3.sde_sample_freq,
                use_sde_at_warmup=cfg.sb3.use_sde_at_warmup,
                create_eval_env=True,
                seed=cfg.sb3.seed,
                device="auto",
                _init_setup_model=True,
                policy_kwargs=dict(net_arch=list(cfg.sb3.hid)),
                tensorboard_log="runs/test",
                # max_episode_length=training_config["algorithm"]["max_ep_len"]
                )
    model.learn(
            total_timesteps=cfg.sb3.n_steps,
            log_interval=cfg.sb3.log_interval,
            reset_num_timesteps=True,
            eval_freq=cfg.sb3.eval_freq,
            n_eval_episodes=cfg.sb3.n_eval_episodes
        )


if __name__ == "__main__":
    main()
