import numpy as np


def l1_distance(x1: np.ndarray, x2: np.ndarray) -> np.int32:
    """
        L1 (Manhattan) distance is just an absolute distance between 2 vectors

        l1_distance = |x1[0] - x2[0]| + |x1[1] - x2[0]| + ...
    """
    return np.sum(np.abs(x1 - x2))

def l1_norm(x: np.ndarray) -> np.int32:
    """
        Just a sum of all absolute values in the input vector

        l1_norm = |x[0]| + |x[1]|
    """
    return np.sum(np.abs(x))
