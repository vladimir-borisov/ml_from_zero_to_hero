import numpy as np
from neural_networks.activation_functions import ReLU, Softmax, ELU, Sigmoid, Linear
from neural_networks.loss_functions import MeanSquaredError, SquareLoss


'''
One layer perceptron: class consists of only 2 layers (input, output)
'''
class OneLayerPreceptron:

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
    def fit_batch(self, x, labels):
        pass

    #update weights by 1 sample
    def fit_one(self, x, labels):
        self.backward(x, labels)


    def forward(self, x):
        weighted_sum = x.dot(self.w) + self.bias
        self.outputs = self.activation_function(weighted_sum)

    def backward(self, x, true_labels):

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
