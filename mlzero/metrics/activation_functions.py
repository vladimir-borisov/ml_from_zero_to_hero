import numpy as np

class Linear:
    def __call__(self, x):
        return x
    def gradient(self, x):
        return 1

class Sigmoid:
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def inverse(self, y):
        """
            Get input value (X) by the function value (Y)

            Input:
                y: the output of the Sigmoid(x), usually treated as probability
            Output:
                x the intput of the Sigmoid(x), usually treated as log(odds)
        """
        return np.log(y / (1 - y))


    def gradient(self, x):
        return np.exp(x) / (1 + np.exp(x))

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

