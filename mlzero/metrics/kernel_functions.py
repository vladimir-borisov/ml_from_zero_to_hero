import numpy as np

""" 
Links:

1. https://scikit-learn.org/stable/modules/svm.html#kernel-functions

"""

def linear():
    def f(x1: np.ndarray, x2: np.ndarray):
        return x1 @ x2.T
    return f

def polynomial(degree: int = 1, gamma: float = 1, coef0: float = 0):
    def f(x1: np.ndarray, x2: np.ndarray):
        return (gamma * (x1 @ x2.T) + coef0) ** degree
    return f

def rbf(gamma: float = 1):
    def f(x1: np.ndarray, x2: np.ndarray):
        l1_distance = np.linalg.norm(x1 - x2, ord = 1)
        l1_distance = l1_distance * l1_distance
        return np.exp(-gamma * l1_distance)
    return f

def sigmoid(degree: int = 1, gamma: float = 1, coef0: float = 0):
    def f(x1: np.ndarray, x2: np.ndarray):
        return (gamma * (x1 @ x2.T) + coef0) ** degree
    return f
