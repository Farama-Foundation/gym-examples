"""This file describes an evaluation callback for Optuna.

Author: Jakob Thumm
Date: 14.10.2022
"""
import optuna
import numpy as np


class TrialEvalCallback():
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        trial: optuna.trial.Trial,
    ):
        """Initialize the callback.

        Args:
            trial (optuna.trial.Trial): Trial to report to.
        """
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.last_mean_reward = -np.inf

    def _on_step(
        self,
        mean_reward: float
    ):
        """Evaluate the model and report the result to the trial.

        Args:
            mean_reward (float): Mean reward of the last evaluation.
        """
        self.eval_idx += 1
        self.last_mean_reward = mean_reward
        self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
        # Prune trial if need
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True
