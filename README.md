# Gym Examples
Some simple examples of Gym environments and wrappers.
For some explanations of these examples, see the [Gym documentation](https://www.gymlibrary.ml).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://www.gymlibrary.ml/pages/environment_creation/index).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://www.gymlibrary.ml/pages/wrappers/index).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/gym-docs).