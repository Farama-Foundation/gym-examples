from gymnasium.envs.registration import register

register(
    id="gym_onkorobot/OnkoRobot-v0",
    entry_point="gym_onkorobot.envs:OnkoRobotEnv",
)
