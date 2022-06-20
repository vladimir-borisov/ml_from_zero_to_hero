import numpy as np
from typing import Tuple

def shuffle(*arrays: Tuple[np.ndarray], random_state: int = None, stratify: bool = True) -> list:
    """
        Randomly shuffle elements in the given arrays

        Input:
            *arrays: list of numpy arrays
            random_state: initial values for random generator
            stratify: if True save the same order between arrays a1[0], a2[0] -> a1[new_idx], a2[new_idx]
                      if False a1[0], a2[0] -> a1[new_idx1], a2[new_idx2]
        Output:
            given arrays with shuffled elements
    """

    idx = np.random.permutation(len(arrays[0]))
    result = []

    for array in arrays:
        if (not stratify):
            idx = np.random.permutation(len(array))
        result.append(array[idx])

    return result


def train_test_split(*arrays) -> tuple:
    """
        Input:
            *arrays: list of numpy arrays
        Output:
            tuple of X, y with randomly shuffled elements
    """
    pass