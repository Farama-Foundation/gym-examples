from rl_zoo3.train import train
import n_dim_reach_env  # noqa: F401
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from n_dim_reach_env.wrappers.HER_buffer_add_monkey_patch import (
    custom_add,
    _custom_sample_transitions,
)

if __name__ == "__main__":  # noqa: C901
    # HerReplayBuffer.add = custom_add
    # HerReplayBuffer._sample_transitions = _custom_sample_transitions
    train()
