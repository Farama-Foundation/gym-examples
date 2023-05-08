from gym_onkorobot.core.observation import Observation
from pandas import DataFrame
from pandera.errors import SchemaError
import numpy as np


def test_dose():
    obs = Observation()

