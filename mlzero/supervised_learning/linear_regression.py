from __future__ import annotations # for self annotation in the return statement
import numpy as np
from typing import Callable
from mlzero.metrics.loss_functions import MeanSquaredError as MSE
from mlzero.metrics.regularization_functions import L1_regularization, L2_regularization, L1_L2_regularization

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
            1. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.htmls
        Notes:
            1. biases are included in X and W matrix

    """

    def __init__(self, loss: Callable = MSE(), max_iter: int = 1000, learning_rate: float = 0.01,
                 regularization_function = None) -> LinearRegressionSGD:
        """

            Input:
                loss: loss function for the model
                max_iter: number of gradient descent updates
                learning_rate: coefficient for gradient scaling during weights updating step
                regularization_function: regularization for weights (e.g. L1, L2)
            Output:
                self
        """
        self.loss = loss
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regularization_function = regularization_function

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

            # add gradient from regularization function if it's defined
            if (self.regularization_function is not None):
                weights_gradient += self.regularization_function.gradient(self.weights)

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


class LassoRegression(LinearRegressionSGD):
    """
        Linear regression with L1 norm which is used as a regularization for weights

        Actually we optimize:

            LOSS_FUNCTION + L1_NORM_OF_WEIGHTS

        Links:
            1.https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    def __init__(self, loss: Callable = MSE(), max_iter: int = 1000, learning_rate: float = 0.01, alpha: float = 1.0) -> LassoRegression:
        """

            Input:
                alpha: regularization strength, bigger -> stronger
            Output:
                -
        """
        super(LassoRegression, self).__init__(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                                              regularization_function=L1_regularization(alpha=alpha))

class RidgeRegression(LinearRegressionSGD):
    """
        Linear regression with L2 norm which is used as a regularization for weights

        Actually we optimize:

            LOSS_FUNCTION + L2_NORM_OF_WEIGHTS

    """

    def __init__(self, loss: Callable = MSE(), max_iter: int = 1000, learning_rate: float = 0.01, alpha: float = 1.0) -> RidgeRegression:
        """

            Input:
               loss: function which we try to minimize
               max_iter: how many iterations we are reade
               learning_rate: coefficient for gradient scaling during weights updating step
               alpha: regularization strength, bigger -> stronger
            Output:

        """
        super(RidgeRegression, self).__init__(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                                              regularization_function=L2_regularization(alpha=alpha))


class ElasticNetRegression(LinearRegressionSGD):
    """
        Linear regression with L1+L2 norm which is used as a regularization for weights

        Actually we optimize:

            LOSS_FUNCTION + L1_NORM_OF_WEIGHTS + L2_NORM_OF_WEIGHTS

    """

    def __init__(self, loss: Callable = MSE(), max_iter: int = 1000, learning_rate: float = 0.01,
                 l1_ratio: float = 0.5, alpha: float = 1.0) -> ElasticNetRegression:
        """

            Input:
               loss: function which we try to minimize
               max_iter: how many iterations we are reade
               learning_rate: coefficient for gradient scaling during weights updating step
               alpha: regularization strength, bigger -> stronger
            Output:

        """
        super(ElasticNetRegression, self).__init__(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                                                   regularization_function=L1_L2_regularization(alpha=alpha, l1_ratio=l1_ratio))
