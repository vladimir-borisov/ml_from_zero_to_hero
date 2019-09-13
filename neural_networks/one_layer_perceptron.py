import numpy as np
from neural_networks.activation_functions import ReLU, Softmax, ELU

'''
One layer perceptron: class consist of only 2 layers (input, output)
'''
class OneLayerPreceptron:
    def __init__(self, input_size, output_size, activation_function = Softmax):
        #set random weights between [0, 1)
        self.w = np.random().rand(input_size, output_size)
        self.bias = np.zeros(output_size)

        #inner variables for outputs and inputs for access after forward and backward operations
        self.outputs = np.zeros(output_size)
        self.inputs = np.zeros(input_size)

        #set activation function
        self.activation_function = activation_function

    #forward + backward
    def fit(self, x, y_true):
        pass

    def forward(self, x):
        weighted_sum = x.dot(self.w * x) + self.bias
        self.outputs = self.activation_function(weighted_sum)

    def predict(self, x):
        self.forward(x)
        return self.outputs


model = OneLayerPreceptron(4, 2)

#TODO fix error with numpy module
model.predict([1, 2, 3, 4])

