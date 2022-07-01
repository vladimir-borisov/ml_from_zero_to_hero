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

def bootstrap(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
        Apply boostrap to the input array + labels
        We randomly take the data from the input array and labels and return it back.

        Output array can have the equal entries because of boostrap logic

        Input:
            X: 2d input array of features
            y: 1d input array of labels for the input samples
        Output
            bootstrapped features and labels
    """

    n_samples = len(y)

    idx = np.random.randint(n_samples, size=n_samples)

    return np.copy(X[idx]), np.copy(y[idx])


def get_most_frequent_value(x: np.ndarray):
    """
        Get the most frequent value in the input array

        Input:
            x: numpy array any size with ALL NON-NEGATIVE values
        Output:
            the most frequent value
    """

    counts = np.bincount(x.flatten())
    return np.argmax(counts)
