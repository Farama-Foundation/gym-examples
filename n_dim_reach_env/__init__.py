from gym.envs.registration import register

register(
    id="ReachEnv-v0",
    entry_point="n_dim_reach_env.envs:ReachEnv",
)
