import numpy as np
from neural_networks.activation_functions import ReLU, Softmax, ELU, Sigmoid
from neural_networks.loss_functions import MeanSquaredError
'''
One layer perceptron: class consist of only 2 layers (input, output)
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

    #forward + backward
    def fit(self, x, y_true):
        pass

    def forward(self, x):
        weighted_sum = x.dot(self.w) + self.bias
        self.outputs = self.activation_function(weighted_sum)

    def backward(self, x, true_labels):
        weighted_input = x.dot(self.w) + self.bias

        predictions = self.activation_function(weighted_input)

        grad = self.loss_function.gradient(predictions, true_labels) * self.activation_function.gradient(weighted_input)

        #error gradients for weights and biases
        #TODO think about more beautiful way how to do matrix multiplication. (x.T) does not turn the matrix :(
        grad_weights = x.reshape((len(x), 1)).dot(grad.reshape((1, len(grad))))
        grad_bias = grad

        #update weights
        self.w = self.w - self.learning_rate * grad_weights
        self.bias = self.bias - self.learning_rate * grad_bias

    def predict(self, x):
        self.forward(x)
        return self.outputs


model = OneLayerPreceptron(5, 3, 0.01)
print(model.predict(np.array([1, 2, 3, 4, 5])))
print(model.backward(np.array([1, 2, 3, 4, 5]), np.array([1,1,1])))

