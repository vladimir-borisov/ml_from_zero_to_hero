import numpy as np

def standard_scale(data: np.ndarray, axis: int = 0):

    mean_ = np.mean(data, axis = axis)
    std_ = np.std(data, axis = axis)

    print(mean_, std_)
    return data - mean_ / std_


X = np.array([[1, 2, 3],
             [1, 1, 1]])

print(X.shape)

print(standard_scale(X, axis = 0))

