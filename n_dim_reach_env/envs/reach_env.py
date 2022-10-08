"""This file describes a simple reaching gym environment.

Author: Jakob Thumm
Date: 8. Oct. 2022
"""
import gym
from gym import spaces
import numpy as np
import pygame
from typing import Dict, List, Optional, Tuple, Union


class ReachEnv(gym.GoalEnv):
    """A simple reaching environment.

    Characteristica:
        - deterministic transition function
        - state limits that lead to collision
        - adjustable dimensionality.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 n_dim: int = 2,
                 max_action: float = 0.05,
                 goal_distance: float = 0.01,
                 done_on_collision: bool = False,
                 randomize_env: bool = True,
                 collision_reward: float = -1.0,
                 goal_reward: float = 1.0,
                 step_reward: float = -1.0,
                 reward_shaping: bool = True,
                 render_mode: bool = None,
                 seed: int = 42):
        """Create the reach environment.

        Args:
            n_dim: Number of dimensions of state and actions.
            max_action: Maximal distance the agent can take in a single step.
            goal_distance: Distance to the goal, so that the goal is reached.
            done_on_collision: Terminate the episode on collision.
            randomize_env: Randomized start and goal position.
            collision_reward: Reward for colliding with the environment edge.
                (gets added to the step reward!)
            goal_reward: Reward for reaching the goal.
                (gets added to the step reward!)
            step_reward: Reward for taking a setp.
            reward_shaping: Set true for dense rewards and false for sparse.
            render_mode: Type of rendering.
            seed: random seed.
        """
        self.n_dim = n_dim
        self.max_action = max_action
        self.goal_distance = goal_distance
        self.done_on_collision = done_on_collision
        self.randomize_env = randomize_env
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.rnd_seed = seed
        if self.render_mode is not None:
            self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-1, 1,
                                          shape=(self.n_dim,), dtype=float,
                                          seed=self.rnd_seed),
                "achieved_goal": spaces.Box(-1, 1,
                                            shape=(self.n_dim,), dtype=float,
                                            seed=self.rnd_seed),
                "desired_goal": spaces.Box(-1, 1,
                                           shape=(self.n_dim,), dtype=float,
                                           seed=self.rnd_seed),
            }
        )
        self.action_space = spaces.Box(-max_action, max_action,
                                       shape=(self.n_dim,), dtype=float,
                                       seed=self.rnd_seed)
        self._agent_location = np.zeros((self.n_dim,))
        self._target_location = np.zeros((self.n_dim,))
        if not self.randomize_env:
            self.start_state = np.zeros((self.n_dim,))
            # We choose random values of -0.5 or 0.5 per dimension.
            np.random.seed(seed)
            self._target_location = (
                1.0 * np.random.randint(2, size=self.n_dim) - 0.5
            )
        self.size = 1
        self.max_distance = np.sqrt(2*(self.size)**2)
        self._collision = False
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is
        used to ensure that the environment is rendered at the correct
        framerate in human-mode. They will remain `None` until human-mode
        is used for the first time.
        """
        assert (render_mode is None or
                render_mode in self.metadata["render_modes"])
        # We can only render dimension 2 right now.
        assert (self.n_dim == 2 or render_mode is None)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self) -> Dict:
        return {
            "observation": self._agent_location,
            "achieved_goal": self._agent_location,
            "desired_goal": self._target_location
        }

    def _get_info(self) -> Dict:
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location
            ),
            "goal_reached": self._get_success(
                self._get_obs()["achieved_goal"],
                self._get_obs()["desired_goal"]),
            "collision": self._collision
        }

    def _get_success(self,
                     achieved_goal: np.ndarray,
                     desired_goal: np.ndarray) -> bool:
        """Get the information if the goal was reached.

        Args:
            achieved_goal: Achieved goal in this step.
            desired_goal: Desired goal in this step.
        Returns:
            True if goal was reached, false otherwise.
        """
        return np.linalg.norm(
                achieved_goal - desired_goal
            ) < self.goal_distance

    def _get_done(self,
                  achieved_goal: np.ndarray,
                  desired_goal: np.ndarray,
                  info: Dict) -> bool:
        """Return if the episode is finished.

        Args:
            achieved_goal: Achieved goal in this step.
            desired_goal: Desired goal in this step.
            info: Information dictionary.
        Returns:
            True if done flag should be set, false otherwise.
        """
        if self.done_on_collision and info["collision"]:
            return True
        return self._get_success(achieved_goal, desired_goal)

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        # We need the following line to seed self.np_random
        super().reset()
        self._collision = False
        # Choose the agent's and goal location.
        if self.randomize_env:
            self._agent_location = self.observation_space["achieved_goal"].sample()  # noqa: E501
            # We will sample the target's location randomly until it does not
            # coincide with the agent's location
            self._target_location = self.observation_space["desired_goal"].sample()  # noqa: E501
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.observation_space["desired_goal"].sample()  # noqa: E501
        else:
            self._agent_location = self.start_state

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self,
             action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for
        calling `reset()` to reset this environment's state.

        Accepts an action and returns a tuple
            (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation.
            reward (float) : amount of reward returned after previous action.
            done (bool): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        self._collision = False
        next_pos = self._agent_location + action
        if np.any(np.abs(next_pos) > self.size):
            self._collision = True
        self._agent_location = np.clip(next_pos, -self.size, self.size)
        observation = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(
            achieved_goal=observation["achieved_goal"],
            desired_goal=observation["desired_goal"],
            info=info)
        done = self.compute_done(
            achieved_goal=observation["achieved_goal"],
            desired_goal=observation["desired_goal"],
            info=info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def compute_reward(self,
                       achieved_goal: Union[List[np.ndarray], np.ndarray],
                       desired_goal: Union[List[np.ndarray], np.ndarray],
                       info: Union[List[Dict], Dict]
                       ) -> Union[List[float], float]:
        """Compute the step reward.

        This externalizes the reward function and makes it dependent
        on an a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent
        of the goal, you can include the necessary values to derive
        it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to
                attempt to achieve.
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal
                w.r.t. to the desired goal.
                Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'], ob['goal'], info)
        """
        distance = np.linalg.norm(achieved_goal-desired_goal, axis=-1)
        goal_reached = distance < self.goal_distance
        if isinstance(info, list):
            collision = np.array([i["collision"] for i in info])
        else:
            collision = info["collision"]
        reward = (np.ones((achieved_goal.shape[0],)) * self.step_reward +
                  collision * self.collision_reward
                  )
        if self.reward_shaping:
            reward += (1 - distance/self.max_distance) * self.goal_reward
        else:
            reward += goal_reached * self.goal_reward
        return reward

    def compute_done(self,
                     achieved_goal: Union[List[np.ndarray], np.ndarray],
                     desired_goal: Union[List[np.ndarray], np.ndarray],
                     info: Union[List[Dict], Dict]
                     ) -> Union[List[bool], bool]:
        """Compute the step reward.

        This externalizes the done function and makes it dependent
        on an a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent
        of the goal, you can include the necessary values to derive
        it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to
                attempt to achieve.
            info (dict): an info dictionary with additional information
        Returns:
            bool: The done flag that corresponds to the provided achieved goal
                w.r.t. to the desired goal.
                Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert done == env.compute_done(
                    ob['achieved_goal'], ob['goal'], info)
        """
        distance = np.linalg.norm(achieved_goal-desired_goal, axis=-1)
        goal_reached = distance < self.goal_distance
        if self.done_on_collision:
            if isinstance(info, list):
                collision = np.array([i["collision"] for i in info])
            else:
                collision = info["collision"]
            done = np.logical_or(goal_reached, collision)
        else:
            done = goal_reached
        return done

    def render(self):
        """Render a frame."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a frame of the current state using rgb_array method."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / (2*self.size)
        )  # The size of a single grid square in pixels
        target = [self._target_location[0] + self.size,
                  self._target_location[1] + self.size]
        agent = [self._agent_location[0] + self.size,
                 self._agent_location[1] + self.size]
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * target,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the
            # visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the
            # predefined framerate.
            # The following line will automatically add a delay to keep
            # the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Close the render window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
