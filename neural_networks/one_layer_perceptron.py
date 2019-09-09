import numpy as np

'''
One layer perceptron: class consist of only 2 layers (input, output)
'''
class OneLayerPreceptron:
    def __init__(self, w_count):
        self.w = np.random().rand(1, w_count)
        self.bias = np.zeros(w_count)

    #forward + backward
    def fit(self):
        pass

    def backward(self):
        pass

    def forward(self):
        pass
