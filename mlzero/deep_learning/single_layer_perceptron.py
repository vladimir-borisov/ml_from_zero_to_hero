import numpy as np
from mlzero.deep_learning.activation_functions import Sigmoid
from mlzero.deep_learning.loss_functions import MeanSquaredError

'''
Signle Layer Perceptron: class consists of only 2 layers (input, output)
'''
class SingleLayerPreceptron:

    def __init__(self, input_size, output_size, learning_rate, activation_function = Sigmoid, loss_function = MeanSquaredError):

        #set random weights between [0, 1)
        self.w = np.random.rand(input_size, output_size)
        self.bias = np.zeros(output_size)

        #inner variables for outputs and inputs for a ccess after forward and backward operations
        self.outputs = np.zeros(output_size)
        self.inputs = np.zeros(input_size)

        #set activation function
        self.activation_function = activation_function()

        #set loss function
        self.loss_function = loss_function()

        #set learning rate
        self.learning_rate = learning_rate

    #update weights by batch
    def fit(self, x, true_labels):
        pass

    '''
        Make the forward pass

        Input:
            x: numpy ndarray
                this value must to have the same size like the network expect
        Output
            None
    '''
    def forward(self, x):
        weighted_sum = x.dot(self.w) + self.bias
        self.outputs = self.activation_function(weighted_sum)

    '''
        The function changes weights by one sample
        
        Input: 
            x: numpy ndarray
                Input values 
            true_labels: numpy ndarray 
                Correct labels for inputs
        Output:
            None
    '''
    def fit_one(self, x, true_labels):

        #Reshape x to shape (1, len(x)) for easy the next opeartions like transpose
        x = x.reshape((1, len(x)))

        #forward
        weighted_input = x.dot(self.w) + self.bias
        predictions = self.activation_function(weighted_input)

        #gradient calculations by chain rule
        grad = self.loss_function.gradient(predictions, true_labels) * self.activation_function.gradient(weighted_input)

        #error gradients for weights and biases
        grad_weights = x.T.dot(grad)
        grad_bias = grad

        #update weights and biases
        self.w = self.w - self.learning_rate * grad_weights
        self.bias = self.bias - self.learning_rate * grad_bias

    def predict(self, x):
        self.forward(x)
        return self.outputs