from __future__ import annotations # for self annotation in the return statement
import numpy as np
from typing import Callable
from mlzero.metrics.loss_functions import MeanSquaredError as MSE


class LinearRegression:
    """
        Linear regression approximate given points (X, y) with
        a linear equation using Ordinary Least Squares method

        y = 1 * w0 + x1 * w1 + ... + xN * wN

        Vector form:

        y = X * W, where X - input variables
                         W - coefficients of Linear Regression (w0, w1, ... , wN)

        Links:
            1. http://aco.ifmo.ru/el_books/numerical_methods/lectures/glava4.html (RU)
            2. https://global.oup.com/booksites/content/0199268010/samplesec3 (ENG)

        Notes:
            1. biases are included in X and W matrix

    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
            Calculate the best coefficients for f(x) -> y

            Input:
                X: input variable with shape (number_sample, number_features)
                y: target values with shape (number_samples, )
            Output:
                self
        """

        # add a fake first column with ones instead of using separate bias variable
        X = np.insert(X, 0, 1, axis=1)

        # trying to minimize the least square loss using the following equation
        self.weights = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predict values for the input features

            Input:
                X: input variable with shape (number_sample, number_features)
            Output:
                predictions for each sample with shape (number_samples, )
        """

        # add a fake first column with ones instead of using separate bias variable
        X = np.insert(X, 0, 1, axis=1)

        return X @ self.weights

class LinearRegressionSGD:
    """
        Linear regression approximate given points (X, y) with
        SGD (Stochastic Gradient Descent)

        y = 1 * w0 + x1 * w1 + ... + xN * wN

        Vector form:

        y = X * W, where X - input variables
                         W - coefficients of Linear Regression (w0, w1, ... , wN)

        Links:
            -

        Notes:
            1. biases are included in X and W matrix

    """

    def __init__(self, loss: Callable = MSE(), max_iter: int = 1000, learning_rate: float = 0.01) -> LinearRegressionSGD:
        """

            Input:
                loss: loss function for the model
                max_iter:
            Output:
                self
        """
        self.loss = loss
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
            Calculate the best coefficients for f(x) -> y

            Input:
                X: input variables with shape (number_samples, number_features)
                y: target values with shape (number_samples, )
            Output:
                self
        """

        # add a fake first column with ones instead of using separate bias variable
        X_ = np.insert(X, 0, 1, axis=1)

        self.weights = np.random.rand(X_.shape[1]) # just a random initialization in [0, 1) values range

        for iter_ind in range(self.max_iter):

            y_pred = self.predict(X)

            loss_gradient = self.loss.gradient(y, y_pred)
            weights_gradient = loss_gradient @ X_  # we use the chain rule to get loss gradient with respect to each weight

            self.weights -= weights_gradient * self.learning_rate

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predict values for the input features

            Input:
                X: input variable with shape (number_sample, number_features)
            Output:
                predictions for each sample with shape (number_samples, )
        """

        # add a fake first column with ones instead of using separate bias variable
        X_ = np.insert(X, 0, 1, axis=1)

        return X_ @ self.weights
