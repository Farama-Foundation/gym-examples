"""This is a script to optimize hyperparameters for the RL agent.

Author: Jakob Thumm
Date: 14.10.2022
"""
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler


def hyperparam_optimization(
    algo,
    model_fn,
    env_fn,
    env_args,
    learn_args,
    n_trials=10,
    n_timesteps=5000,
    hyperparams=None,
    n_jobs=1,
    sampler_method='random',
    pruner_method='halving',
    seed=0,
    verbose=1):
    """
    Optimize hyperparameters using Optuna.
    :param algo: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param env_args: (dict) Arguments for env fun
    :param env_args: (dict) Arguments for model.learn() fn
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict)
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    if hyperparams is None:
        hyperparams = {}

    n_startup_trials = 10
    # test during 5 episodes
    n_eval_episodes = 5
    # evaluate every 20th of the maximum budget per iteration
    n_evaluations = 20
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
    algo_sampler = HYPERPARAMS_SAMPLER[algo]

    def objective(trial):

        kwargs = hyperparams.copy()

        trial.model_class = None
        if algo == 'her':
            trial.model_class = hyperparams['model_class']

        # Hack to use DDPG/TD3 noise sampler
        if algo in [Algorithm.TD3] or trial.model_class in [Algorithm.TD3]:
            trial.n_actions = 2#env_fn(n_envs=1).action_space.shape[0]
        kwargs.update(algo_sampler(trial))

        model = model_fn(env = env_fn(**env_args), **kwargs)

        eval_env = env_fn(**env_args)
        # Account for parallel envs
        eval_freq_ = eval_freq
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=n_eval_episodes,
                                          eval_freq=eval_freq_, deterministic=True)

        try:
            model.learn(callback=eval_callback,
                        total_timesteps=n_timesteps,
                        **learn_args)
            # Free memory
            #model.env.close()
            eval_env.close()
        except:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            #model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        cost = -1 * eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost

    if algo in PRIOR_KNOWLEDGE:
        study.enqueue_trial(PRIOR_KNOWLEDGE[algo]())
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
