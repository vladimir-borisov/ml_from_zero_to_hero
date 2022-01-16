class StochasticGradientDescent:
    def __init__(self):
        self.learning_rate = 0.01

    #calculate and return the updated weights
    def update(self, w, gradient):
        return  w - gradient * self.learning_rate