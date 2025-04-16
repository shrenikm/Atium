import numpy as np

from common.custom_types import AngleOrAnglesRad


def normalize_angle(angles: AngleOrAnglesRad) -> AngleOrAnglesRad:
    # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    return (np.array(angles) + np.pi) % (2 * np.pi) - np.pi
