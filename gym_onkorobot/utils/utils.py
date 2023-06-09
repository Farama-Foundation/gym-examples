
from random import random


def generate_infection(infection_percent = 0.2):
    p = random()
    return 1 if p > (1-infection_percent) else 0
