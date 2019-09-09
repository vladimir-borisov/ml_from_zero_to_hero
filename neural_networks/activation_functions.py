import numpy as np

'''
    Linear activation function like (kx + b)

    Params:
    x = variable
    k = angle coefficient
    b = bias coefficient
'''
def Linear (x, k, b):
    return k * x + b


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

