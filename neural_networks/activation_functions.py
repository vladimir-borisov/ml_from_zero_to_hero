import numpy as np

def Linear (x, k, b):
    return k * x + b

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def Softmax(x):
    exps = np.exp(x)
    sum = exps.sum()
    return exps / sum

def Tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

def ReLU(x):
    if (x <= 0):
        return 0
    else:
        return x

def LeakyReLU(x):
    if (x < 0):
        return 0.01 * x
    else:
        return x

#optimal value of a ~ [0.01 - 0.05]
def PReLU (x, a):
    if (x < 0):
        return a * x
    else:
        return x

def ELU (x, a):
    if (x < 0):
        return a * (np.exp(a) - 1)
    else:
        return x


