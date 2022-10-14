"""This is a script to optimize hyperparameters for the RL agent.

Author: Jakob Thumm
Date: 14.10.2022
"""
from typing import Any, Dict, List
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler

from n_dim_reach_env.rl.train_droq import train_droq
from n_dim_reach_env.rl.callbacks.trial_eval_callback import TrialEvalCallback

from n_dim_reach_env.conf.config_droq import EnvConfig


def optimize_hyperparameters(
    env_fn: callable,
    env_args: EnvConfig,
    obs_space_fn: callable,
    has_dict_obs_fn: callable,
    learn_args: Dict[str, Any],
    tuning_params: List[str],
    n_trials: int = 10,
    n_startup_trials: int = 5,
    n_timesteps: int = 100000,
    n_jobs: int = 1,
    sampler_method: str = 'random',
    pruner_method: str = 'halving',
    n_eval_episodes: int = 5,
    n_evaluations: int = 20,
    seed: int = 0,
    use_prior: bool = False,
    verbose: int = 1
):
    """Optimize hyperparameters using Optuna.

    :param env_fn: (func) function that is used to instantiate the env
    :param env_args: (EnvConfig) Arguments for env fun
    :param obs_space_fn: (func) function that is used to instantiate the observation space
    :param has_dict_obs_fn: (func) function that outputs whether the env has a dict obs space
    :param learn_args: (dict) Arguments for training fn
    :param tuning_params: (list) List of hyperparams to tune
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_startup_trials: (int) number of trials before using the sampler
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param n_eval_episodes: (int) number of episodes to evaluate the agent
    :param n_evaluations: (int) number of evaluations per trial
    :param seed: (int)
    :param use_prior: (bool) whether to use prior knowledge for DroQ hyperparams
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    eval_freq = int(n_timesteps / n_evaluations)

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == 'skopt':
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(skopt_kwargs={
            'base_estimator': "GP",
            'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler_method))

    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials,
                              n_warmup_steps=n_evaluations // 3)
    elif pruner_method == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials,
                              n_warmup_steps=n_evaluations)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner_method))

    if verbose > 0:
        print("Sampler: {} - Pruner: {}".format(sampler_method, pruner_method))

    study = optuna.create_study(sampler=sampler, pruner=pruner)

    def objective(trial):
        eval_callback = TrialEvalCallback(trial)
        kwargs = learn_args.copy()
        kwargs.update(sample_droq_params(trial, tuning_params))
        kwargs.update({'eval_callback': eval_callback._on_step,
                       'max_steps': n_timesteps,
                       'eval_interval': eval_freq,
                       'eval_episodes': n_eval_episodes})
        # Hack to use DDPG/TD3 noise sampler
        # if algo in [Algorithm.TD3] or trial.model_class in [Algorithm.TD3]:
        #     trial.n_actions = 2#env_fn(n_envs=1).action_space.shape[0]
        # kwargs.update(algo_sampler(trial))
        eval_env = env_fn(env_args)
        observation_space = obs_space_fn(eval_env)
        dict_obs = has_dict_obs_fn(eval_env)
        # Account for parallel envs
        try:
            train_droq(
                env=eval_env,
                observation_space=observation_space,
                dict_obs=dict_obs,
                **kwargs
            )
            # Free memory
            eval_env.close()
        except Exception as e:
            print(e)
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        cost = -1 * eval_callback.last_mean_reward

        del eval_env

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost

    if use_prior:
        study.enqueue_trial(prior_droq_params(tuning_params))
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study


def sample_droq_params(
    trial: optuna.trial.Trial,
    tuning_params: List[str],
) -> Dict[str, Any]:
    """
    Sampler for DroQ hyperparams.

    Args:
        trial: Optuna trial
        tuning_params: List of hyperparams to tune
    Returns:
        Dict of sampled hyperparams
    """
    hyperparams = dict()
    hyperparams['agent_kwargs'] = dict()
    if 'actor_lr' in tuning_params:
        actor_lr = trial.suggest_loguniform('actor_lr', 1e-6, 0.01)
        hyperparams['agent_kwargs']['actor_lr'] = actor_lr
    if 'critic_lr' in tuning_params:
        critic_lr = trial.suggest_loguniform('critic_lr', 1e-6, 0.01)
        hyperparams['agent_kwargs']['critic_lr'] = critic_lr
    if 'temp_lr' in tuning_params:
        temp_lr = trial.suggest_loguniform('temp_lr', 1e-6, 0.01)
        hyperparams['agent_kwargs']['temp_lr'] = temp_lr
    if 'hidden_dims' in tuning_params:
        hidden_dims = trial.suggest_categorical('hidden_dims', [
            [64, 64],
            [128, 128],
            [256, 256],
            [64, 64, 64],
            [128, 128, 128],
            [256, 256, 256],
        ])
        hyperparams['agent_kwargs']['hidden_dims'] = hidden_dims
    if 'discount' in tuning_params:
        discount = trial.suggest_categorical('discount', [0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
        hyperparams['agent_kwargs']['discount'] = discount
    if 'tau' in tuning_params:
        tau = trial.suggest_categorical('tau', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]) 
        hyperparams['agent_kwargs']['tau'] = tau
    if 'num_qs' in tuning_params:
        num_qs = trial.suggest_categorical('num_qs', [1, 2, 3]) 
        hyperparams['agent_kwargs']['num_qs'] = num_qs
    # num_min_qs: null
    if 'critic_dropout_rate' in tuning_params:
        critic_dropout_rate = trial.suggest_loguniform('critic_dropout_rate', 1e-4, 0.1)
        hyperparams['agent_kwargs']['critic_dropout_rate'] = critic_dropout_rate
    if 'critic_layer_norm' in tuning_params:
        critic_layer_norm = trial.suggest_categorical('critic_layer_norm', [True, False])
        hyperparams['agent_kwargs']['critic_layer_norm'] = critic_layer_norm
    if 'target_entropy' in tuning_params:
        target_entropy = trial.suggest_categorical('target_entropy', [-0.1, -0.5, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        hyperparams['agent_kwargs']['target_entropy'] = target_entropy
    if 'init_temperature' in tuning_params:
        init_temperature = trial.suggest_loguniform('init_temperature', 1e-4, 0.5)
        hyperparams['agent_kwargs']['init_temperature'] = init_temperature
    if 'sampled_backup' in tuning_params:
        sampled_backup = trial.suggest_categorical('sampled_backup', [True, False])
        hyperparams['agent_kwargs']['sampled_backup'] = sampled_backup
    # buffer_size: 1000000
    # use_her: true
    # n_her_samples: 4
    # goal_selection_strategy: future
    # handle_timeout_termination: true
    # start_steps: 2000
    if 'batch_size' in tuning_params:
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        hyperparams['batch_size'] = batch_size
    if 'utd_ratio' in tuning_params:
        utd_ratio = trial.suggest_categorical('utd_ratio', [1, 2, 5, 10, 20, 40])
        hyperparams['utd_ratio'] = utd_ratio
    return hyperparams


def prior_droq_params(tuning_params: List[str]) -> Dict[str, Any]:
    """
    Prior knowledge for DroQ hyperparams.

    :param tuning_params: List of hyperparams to tune

    :return: (dict)
    """
    return {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "temp_lr": 3e-4,
        "hidden_dims": [256, 256],
        "discount": 0.99,
        "tau": 0.005,
        "num_qs": 2,
        "critic_dropout_rate": 0.01,
        "critic_layer_norm": True,
        "target_entropy": -1,
        "init_temperature": 0.1,
        "sampled_backup": True,
        "buffer_size": 1000000,
        "n_her_samples": 4,
        "goal_selection_strategy": "future",
        "handle_timeout_termination": True,
        "start_steps": 2000,
        "batch_size": 128,
        "utd_ratio": 20,
    }