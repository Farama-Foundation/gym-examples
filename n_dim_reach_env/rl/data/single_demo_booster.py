"""This file describes a single demo booster for DroQ with HER."""

from dataclasses import dataclass
from typing import List
import gym
import numpy as np
from flax.core import frozen_dict
from n_dim_reach_env.rl.data.dataset import DatasetDict

from n_dim_reach_env.rl.data.replay_buffer import ReplayBuffer
from n_dim_reach_env.rl.util.dict_conversion import single_obs


@dataclass
class Transition:
    """A single transition in the replay buffer."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict


@dataclass
class Demonstration:
    """A single demonstration."""

    length: int
    transitions: List[Transition]


def create_single_demo(env: gym.Env) -> Demonstration:
    """Create a single human demonstration.

    Args:
        env (gym.Env): Environment.

    Returns:
        Demonstration: Single human demonstration.
    """
    demo = Demonstration(length=0, transitions=[])
    demo_length = -1
    while demo_length < 5:
        observation, done = env.reset(), False
        desired_goal = observation['desired_goal']
        current_pos = observation['achieved_goal']
        demo_length = np.ceil(np.max(np.abs(current_pos - desired_goal)/np.abs(env.action_space.high)))
    start = current_pos
    goal = desired_goal
    delta = (goal - start)/demo_length
    for _ in range(int(demo_length)):
        desired_goal = observation['desired_goal']
        current_pos = observation['achieved_goal']
        action = np.clip(delta, env.action_space.low, env.action_space.high)
        next_observation, reward, done, info = env.step(action)
        demo.transitions.append(Transition(
            obs=observation,
            action=action,
            reward=reward,
            next_obs=next_observation,
            done=done,
            info=info
        ))
        demo.length += 1
        observation = next_observation
    return demo


class SingleDemoBooster(ReplayBuffer):
    """Define a Replay Buffer that includes artificial human demonstrations.

    The key idea is to create multiple artificial human demonstrations from a single human demonstration.
    The self replay buffer is used to store the artificial human demonstrations.
    self.replay_buffer is used to store the original data.
    """

    def __init__(
        self,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        observation_space: gym.Space,
        action_space: gym.Space,
        single_demo: Demonstration = None,
        n_artificial_demonstrations: int = 100,
        human_demo_rate: float = 0.25,
        ou_mean: float = 0.0,
        ou_sigma: float = 0.01,
        ou_theta: float = 0.15,
        ou_dt: float = 1e-2,
        proportional_constant: float = 0.1
    ):
        """Initialize the single demonstration booster.

        Args:
            env (gym.Env): Environment.
            replay_buffer (ReplayBuffer): Replay buffer.
            observation_space (gym.Space): Observation space.
            action_space (gym.Space): Action space.
            single_demo (Demonstration, optional): Single human demonstration. Defaults to None.
            n_artificial_demonstrations (int, optional): Number of artificial demonstrations. Defaults to 100.
            human_demo_rate (float, optional): Rate of human demonstrations. Defaults to 0.25.
            ou_mean (float, optional): Ornstein-Uhlenbeck mean. Defaults to 0.0.
            ou_sigma (float, optional): Ornstein-Uhlenbeck sigma. Defaults to 0.2.
            ou_theta (float, optional): Ornstein-Uhlenbeck theta. Defaults to 0.15.
            ou_dt (float, optional): Ornstein-Uhlenbeck dt. Defaults to 1e-2.
            proportional_constant (float, optional): Proportional constant of the controller. Defaults to 0.1.
        """
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Observation space must be a dictionary.'
        render_mode = env.render_mode
        env.render_mode = None
        if single_demo is not None:
            self.single_demo = single_demo
        else:
            self.single_demo = create_single_demo(env)
        capacity = self.single_demo.length * n_artificial_demonstrations
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            next_observation_space=observation_space
        )
        self.replay_buffer = replay_buffer
        self.n_artificial_demonstrations = n_artificial_demonstrations
        self.human_demo_rate = human_demo_rate
        self.ou_mean = np.full(env.action_space.shape, ou_mean)
        self.ou_sigma = ou_sigma
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.noise_prev = np.zeros(self.ou_mean.shape)
        self.proportional_constant = proportional_constant
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Create artificial demonstrations from the single demonstration.
        self.create_artificial_demonstrations(env)
        env.render_mode = render_mode

    def initialize_artificial_trajectory(
        self,
        p_gen_start: np.ndarray,
        p_gen_goal: np.ndarray
    ) -> int:
        """Initialize an artificial trajectory.

        Args:
            p_gen_start (np.ndarray): Start position of the trajectory to generate.
            p_gen_goal (np.ndarray): Goal position of the trajectory to generate.

        Returns:
            length (int): Length of the trajectory to generate.
        """
        self.noise_prev = np.zeros(self.ou_mean.shape)
        p_rec_start = self.single_demo.transitions[0].obs["achieved_goal"]
        p_rec_goal = self.single_demo.transitions[-1].next_obs["achieved_goal"]
        max_dist_rec = np.max(np.abs(p_rec_goal - p_rec_start))
        # a = (p_{gen, goal} - p_{gen_start})/(p_{rec, goal} - p_{rec_start})
        if np.linalg.norm(p_rec_goal - p_rec_start) < 1e-6:
            self.a = 1
        else:
            self.a = (p_gen_goal - p_gen_start) / (p_rec_goal - p_rec_start)
        # Scale the number of time steps
        # The of the trajectories depends on the longest distance in either direction.
        max_dist_gen = np.max(np.abs(p_gen_goal - p_gen_start))
        self.n_steps = int(self.single_demo.length * max_dist_gen/max_dist_rec) + 1
        self.b = p_gen_start - self.a * p_rec_start
        return self.n_steps

    def get_artificial_action(
        self,
        step: int,
        p_meas: np.ndarray
    ) -> np.ndarray:
        """Get an artificial action.

        Args:
            step (int): Current step.
            p_meas (np.ndarray): Measured position.

        Returns:
            np.ndarray: Artificial action.
        """
        rec_idx = int(step / self.n_steps * self.single_demo.length)
        p_rec = self.single_demo.transitions[rec_idx].next_obs["achieved_goal"]
        p_gen = self.a * p_rec + self.b
        action = self.proportional_constant * (p_gen - p_meas)
        action = self.add_ou_noise(action)
        action = np.clip(action, self.action_low, self.action_high)
        return action

    def create_artificial_demonstrations(self, env: gym.Env):
        """Create artificial demonstrations from the single demonstration.
        
        Args:
            env (gym.Env): Environment.
        """
        for _ in range(self.n_artificial_demonstrations):
            observation, done = env.reset(), False
            p_gen_goal = observation['desired_goal']
            p_gen_start = observation['achieved_goal']
            self.initialize_artificial_trajectory(p_gen_start, p_gen_goal)
            for i in range(self.n_steps):
                action = self.get_artificial_action(i, observation['achieved_goal'])
                next_observation, reward, done, info = env.step(action)
                super().insert(
                    dict(observations=single_obs(observation),
                         actions=action,
                         rewards=reward,
                         masks=not done,
                         dones=done,
                         next_observations=single_obs(next_observation)))
                observation = next_observation

    def add_ou_noise(self, action: np.ndarray) -> np.ndarray:
        """Add Ornstein-Uhlenbeck noise to the action.

        Args:
            action (np.ndarray): Action.

        Returns:
            np.ndarray: Action with noise.
        """
        noise = self.noise_prev + self.ou_theta * (self.ou_mean - self.noise_prev) * self.ou_dt +\
            self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=self.ou_mean.shape)
        self.noise_prev = noise
        return action + noise

    def insert(
        self,
        data_dict: DatasetDict,
        **kwargs
    ):
        """Insert data into the replay buffer."""
        self.replay_buffer.insert(data_dict, **kwargs)

    def sample(
        self,
        batch_size: int,
        **kwargs
    ) -> frozen_dict.FrozenDict:
        """Sample data from the replay buffer."""
        human_batch = super().sample(
            int(batch_size * self.human_demo_rate)
        ).unfreeze()
        normal_batch = self.replay_buffer.sample(
            int(batch_size * (1 - self.human_demo_rate)),
            **kwargs
        ).unfreeze()
        merged_batch = {key: np.append(human_batch.get(key, []), normal_batch.get(key, []), axis=0)
                        for key in set(list(human_batch.keys())+list(normal_batch.keys()))}
        return frozen_dict.freeze(merged_batch)
