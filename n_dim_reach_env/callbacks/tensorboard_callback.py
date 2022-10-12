"""This file defines the logging functionality via tensorboard for wandb.

Further defines model saving and loading.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    2.5.22 JT Formatted docstrings
"""
from stable_baselines3.common.utils import safe_mean

from wandb.integration.sb3 import WandbCallback
import wandb

from typing import List, Literal, Optional


class TensorboardCallback(WandbCallback):
    """Custom callback for plotting additional values in tensorboard.

    Args:
        eval_env: The evaluation environment.
        verbose: Extra terminal outputs.
        model_save_path: Path to save the model regularly.
        model_save_freq: Save the model every x episodes.
        gradient_save_freq: Save the gradients every x episodes.
        save_freq: Save the model and replay buffer every x episodes.
        model_file: predefined model file for loading / saving.
        start_episode: Define start episode (if model is loaded).
        additional_log_info_keys: Additionally log these keys from the info.
        n_eval_episodes: Number of evaluation episodes.
        deterministic: No noise on action.
        log_interval: Log every n-th episode.
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        additional_log_info_keys: List[str] = ["goal_reached"],
        log_interval: int = 1,
    ):  # noqa: D107
        super(TensorboardCallback, self).__init__(
            verbose, model_save_path, model_save_freq, gradient_save_freq
        )
        self.additional_log_info_keys = additional_log_info_keys
        self._info_buffer = dict()
        for key in additional_log_info_keys:
            self._info_buffer[key] = []
        self.log_interval = log_interval
        self.episode_counter = 0

    def _on_step(self) -> None:
        if self.locals["dones"][0]:
            for key in self.additional_log_info_keys:
                if key in self.locals["infos"][0]:
                    self._info_buffer[key].append(self.locals["infos"][0][key])
            if (self.episode_counter + 1) % self.log_interval == 0:
                for key in self.additional_log_info_keys:
                    if key in self.locals["infos"][0]:
                        self.logger.record(
                            "rollout/{}".format(key),
                            safe_mean(self._info_buffer[key])
                        )
                        self._info_buffer[key] = []
        super()._on_step()

    def _on_rollout_end(self) -> None:
        """After each n-th rollout (episode), save model."""
        self.episode_counter += 1
        super()._on_rollout_end()
