import numpy as np


class L1_regularization:
    """
        L1 regularization is just a sum of absolute values of all elements in a vector

        L1(x) = |x1| + |x2| + ... + |xN|
    """

    def __init__(self, alpha: float = 1.0):
        """

            Input:
                alpha: regularization strength bigger -> stronger
            Output
                -
        """
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.float32:
        """
            Just a sum of all absolute values in the input vector

            l1_norm = |x[0]| + |x[1]|
        """
        return np.sum(np.abs(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
            Get gradient for L1 regularization

            Input:
                x: input vector of variables with any shape
            Output:
                gradient (sign x) with respect to each element (the same shape as the input vector)
        """

        return self.alpha * np.sign(x)

class L2_regularization:
    """
        L2 (Tikhonov) regularization is a sum of squared values of all elements in a vector

        L2(x) = x1^2 + x2^2 + ... + xN^2
    """

    def __init__(self, alpha: float = 1.0):
        """
            Input:
                alpha: regularization strength bigger -> stronger
            Output
                -
        """
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.float32:
        """
            Calculate L2 regularization for the given vector

            l1_norm = |x[0]| + |x[1]|

            Input:
                x: input vector of variables with any shape
            Output
                L2 regularization - sum of squared elemtns in the input vector

        """
        return self.alpha * np.sum(x * x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
            Get gradient for L1 norm

            Input:
                x: input vector of variables with any shape
            Output:
                L2 regularization gradient with respect to each element in the input vector
        """

        return self.alpha * x



class L1_L2_regularization:

    """
        Combination of L1 and L2 regularization

        With the following formula as for ElasticNet Regression:

        l1_l2_regularization = alpha * l1_ratio * l1_regularization + 0.5 * alpha * (1 - l1_ratio) * l2_regularization
    """


    def __init__(self, alpha: float = 1.0, l1_ratio = 0.5):
        """
            Input:
                alpha: regularization strength bigger -> stronger
                l1_ratio: l1_ratio = 0 the regularization is equal to l2
                          l1_ratio = 1 the regularization is equal to l1
            Output
                -
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.l1_regularization = L1_regularization(alpha = alpha)
        self.l2_regularization = L2_regularization(alpha = alpha)


    def __call__(self, x: np.ndarray) -> np.float32:
        """
            Calculate combined l1 l2 regularization

            Input:
                x: input vector
            Output:
                one number - l1 l2 combined regularization value
        """

        l1_value = self.l1_ratio * self.l1_regularization(x)
        l2_value = 0.5 * (1.0 - self.l1_ratio) * self.l2_regularization(x)

        return l1_value + l2_value

    def gradient(self, x: np.ndarray) -> np.float32:
        """
            Calculate gradient for l1 l2 regularization

            Input:
                x: input vector
            Output:
                vector of gradient with respect to each input value
        """

        l1_gradient = self.l1_ratio * self.l1_regularization.gradient(x)
        l2_gradient = 0.5 * (1.0 - self.l1_ratio) * self.l2_regularization.gradient(x)

        return l1_gradient + l2_gradient
