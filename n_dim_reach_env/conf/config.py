"""Defines the dataclasses of the config files."""
from dataclasses import dataclass
from typing import List, Union


@dataclass
class SAC:
    """SB3 SAC config."""

    seed: int
    hid: List[int]
    n_steps: int
    replay_size: int
    gamma: 0.99
    tau: 0.005
    lr: 0.0005
    batch_size: int
    start_steps: int
    update_every: int
    gradient_steps: int
    action_noise: float
    ent_coef: Union[str, float]
    target_entropy: Union[str, float]
    use_sde: bool
    sde_sample_freq: int
    target_update_interval: int
    use_sde_at_warmup: bool
    num_test_episodes: int
    max_ep_len: int
    save_freq: int
    test_only: bool
    load_episode: int
    run_id: str
    log_interval: int
    eval_freq: int
    n_eval_episodes: int


@dataclass
class Env:
    """Environment config."""

    id: str
    n_dim: int
    max_action: float
    goal_distance: float
    done_on_collision: bool
    randomize_env: bool
    collision_reward: float
    goal_reward: float
    step_reward: float
    reward_shaping: bool
    render_mode: bool
    seed: int


@dataclass
class SACTrainingConfig:
    """Training config."""

    sb3: SAC
    env: Env
    verbose: bool
