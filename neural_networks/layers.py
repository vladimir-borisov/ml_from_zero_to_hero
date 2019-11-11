class Layer:
    def __init__(self):
        self.trainable = True

    def forward(self):
        pass

    def backward(self):
        pass

class Dense(Layer):
    def __init__(self, input_shape = (16, )):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Activation(Layer):
    def __init__(self):
        pass

