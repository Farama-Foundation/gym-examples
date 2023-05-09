from gymnasium.spaces import Space


class MissionSpace(Space[str]):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _gen_mission():
        return "Heal all infected points."

    def contains(self, x: str):
        return True