from neural_networks.loss_functions import MeanSquaredError
from neural_networks.optimizers import StochasticGradientDescent

class NeuralNetwork:
    def __init__(self, loss_function = MeanSquaredError, optimizer = StochasticGradientDescent):

        #layers is a just python list
        self.layers = []

        #loss function
        self.loss_function = loss_function

        #optimizer
        self.optimizer = optimizer


    def add_layer(self, layer):
        pass