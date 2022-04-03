import numpy as np

def manhattan_distance(x1: np.ndarray, x2: np.ndarray):
    """
        Manhattan distance

        dist = |x1[0] - x2[0]| + |x1[1] + x2[1]| + ...
    """
    return np.sum(np.abs(x1 - x2))


def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    """
        Euclidean distance

        dist = sqrt( (x1[0] - x2[0])^2  + (x1[1] - x2[1])^2 + ...)

        Input:
            x1:
            x2:
    """

    return np.sqrt(np.sum(np.square(x1 - x2)))
