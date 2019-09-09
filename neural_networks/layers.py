class Layer:
    def __init__(self):
        self.trainable = True

    def forward(self):
        pass

    def backward(self):
        pass

class Activation(Layer):
    def __init__(self):
        pass

