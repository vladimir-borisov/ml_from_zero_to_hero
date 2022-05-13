import numpy as np

class SquareLoss:
    def __call__(self, y_pred, y):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y_pred, y):
        return -(y - y_pred)

class MeanSquaredError:
    """
        Mean Squared Errors (MSE) loss
    """

    @staticmethod
    def check_vectors_(y_true: np.ndarray, y_pred: np.ndarray):
        """
            Check vectors for equal size and not emptiness

            Input:
                y_true: correct (target) values
                y_pred: predicted values
            Output
                None
        """
        assert (len(y_true) == len(y_pred))
        assert (len(y_true) > 0)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
            Calculate MSE

            Input:
                y_true: 1d numpy array, correct (target) values
                y_pred: 1d nupmy array, predicted values
            Output:
                One number - MSE score
        """

        assert (len(y_true) == len(y_pred))
        assert (len(y_true) > 0)

        return np.mean(np.square((y_true - y_pred)), axis=-1)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
            Calculate derivative of MSE with respect to y_pred vector

            Input:
                y_true: correct (target) values
                y_pred: predicted values
            Output:
                vector of derivatives with respect to each y_pred
        """

        self.check_vectors_(y_true, y_pred)

        return -2.0 * (y_true - y_pred)

class HingeLoss:
    """
        Hinge Loss is used for "maximum-margin" classification in SVM
    """

    def __init__(self):
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
            Calculate Hinge Loss

            Input:
                y_true: correct (target) values. y_true = -1 or 1
                y_pred: predicted values
            Output:
                One number - mean value of hinge loss between each pair of (y_true, y_pred)

        """

        return np.mean(np.maximum(0, 1.0 - y_true * y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
            Calculate derivative of Hinge Loss with respect to y_pred vector

            Input:
                y_true: correct (target) values/ y_true = -1 or 1
                y_pred: predicted values
            Output:
                vector of derivatives with respect to each y_pred
        """

        return -y_true * np.array((1.0 - y_true * y_pred) > 0, dtype=np.int32)
