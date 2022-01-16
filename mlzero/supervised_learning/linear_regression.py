from __future__ import annotations # for self annotation in the return statement
import numpy as np


class LinearRegression:
    """
        Linear regression approximate given points (X, y) with
        linear equation using Ordinary Least Squares (Square Error) formula

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
        self.weights_ = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predict values by the given features
        """

        # add a fake first column with ones instead of using separate bias variable
        X = np.insert(X, 0, 1, axis=1)

        return X @ self.weights_
