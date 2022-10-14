"""Defines the dataclasses of the config files."""
from dataclasses import dataclass
from typing import List


@dataclass
class DroQ:
    """DroQ + HER config."""

    actor_lr: float
    critic_lr: float
    temp_lr: float
    hidden_dims: List[int]
    discount: float
    tau: float
    num_qs: int
    num_min_qs: int
    critic_dropout_rate: float
    critic_layer_norm: bool
    target_entropy: float
    init_temperature: float
    sampled_backup: bool
    buffer_size: int
    use_her: bool
    n_her_samples: int
    goal_selection_strategy: str
    handle_timeout_termination: bool
    start_steps: int
    batch_size: int
    utd_ratio: int


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
    max_ep_len: int
    replace_type: int
    n_resamples: int
    punishment: int


@dataclass
class Train:
    """Training settings."""

    max_steps: int
    eval_interval: int
    eval_episodes: int
    tqdm: bool
    use_wandb: bool
    wandb_project: str
    load_episode: int
    run_id: str


@dataclass
class DroQTrainingConfig:
    """Training config."""

    droq: DroQ
    env: Env
    train: Train
    verbose: bool
