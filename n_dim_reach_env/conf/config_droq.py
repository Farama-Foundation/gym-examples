"""Defines the dataclasses of the config files."""
from dataclasses import dataclass
from pickletools import optimize
from tabnanny import verbose
from typing import List


@dataclass
class DroQConfig:
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
class EnvConfig:
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
class TrainConfig:
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
class OptimizeConfig:
    """Optimization settings."""

    optimize: bool
    tuning_params: List[str]
    n_trials: int
    n_startup_trials: int
    n_timesteps: int
    n_jobs: int
    sampler_method: str
    pruner_method: str
    n_eval_episodes: int
    n_evaluations: int
    seed: int
    use_prior: bool


@dataclass
class DroQTrainingConfig:
    """Training config."""

    droq: DroQConfig
    env: EnvConfig
    train: TrainConfig
    optimize: OptimizeConfig
    verbose: bool
