import numpy as np


def com2sys(filepath):
    return {y: x for x, y in np.loadtxt(filepath, dtype=str)}


def sys2com(filepath):
    return {x: y for x, y in np.loadtxt(filepath, dtype=str)}
