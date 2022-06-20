import numpy as np

class SquaredError:
    """
        Squared Error is a sum of squared residuals between predicted and target values

        Squared Error also known as L2 loss and it's almost the same as MSE loss

        Links:
            1. https://datascience.stackexchange.com/questions/26180/l2-loss-vs-mean-squared-loss [EN]
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
            Calculate Squared Error as a sum of residuals between prediction and target

            Input:
                y_true: 1d numpy array, correct (target) values
                y_pred: 1d numpy array, predicted values
            Output:
                One number - sum of squared residual between target and predicted values
        """

        # 0.5 is added here for getting a good gradient form
        return np.sum(0.5 * np.square(y_true - y_pred))

    def optimal_value(self, y_true: np.ndarray) -> list:
        """
            Find the best approximation when we have only y_true and don't know y_prep

            Since we know that this function is a convex function and
            it has only 1 extremum and this extremum is the global minimum -> we can find this point from the:

            gradient(y_true, y_pred) = 0 -> we can find the optimal y_pred

            Input:
                y_true: array of target binary labels (0 or 1)
            Output:
                mean of y_true - optimal value for prediction
        """

        return np.mean(y_true)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
            Calculate derivative of Squared Error with respect to y_pred
            This gradient is also called


            Input:
                y_true: 1d numpy array, correct (target) values
                y_pred: 1d numpy array, predicted values
            Output:
                vector of derivatives with respect to each y_pred
        """

        return -(y_true - y_pred)

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
                y_pred: 1d numpy array, predicted values
            Output:
                One number - MSE score
        """

        assert (len(y_true) == len(y_pred))
        assert (len(y_true) > 0)

        return np.mean(np.square((y_true - y_pred)), axis=-1)

    # TODO: it's for Squared Error  for MSE it should be different !!!
    def optimal_value(self, y_true: np.ndarray) -> list:
        """
            Find the best approximation when we have only y_true and don't know y_prep

            Since we know that this function is a convex function and
            it has only 1 extremum and this extremum is the global minimum -> we can find this point from the:

            gradient(y_true, y_pred) = 0 -> we can find the optimal y_pred

            Input:
                y_true: array of target binary labels (0 or 1)
            Output:
                mean of y_true - optimal value for prediction
        """

        return np.mean(y_true)

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

        return -2.0 * (y_true - y_pred) / len(y_true)

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


class BinaryCrossEntropy:

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
            Binary Cross Entropy loss

            Input:
                y_true: correct (target) values. y_true = -1 or 1
                y_pred: predicted values
            Output:
                One number - mean value of binary cross entropy loss for each pair of (y_true, y_pred)
        """

        # avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    def optimal_value(self, y_true: np.ndarray) -> list:
        """
            Find the best approximation when we have only y_true and don't know y_prep

            Since we know that this function is a convex function and
            it has only 1 extremum and this extremum is the global minimum -> we can find this point from the:

            gradient(y_true, y_pred) = 0 -> we can find the optimal y_pred

            Input:
                y_true: array of target binary labels (0 or 1)
            Output:
                [predicted class, probability, log(odds)]
        """

        y_unique, counts = np.unique(y_true, return_counts=True)

        # best class is just a major class
        best_class = np.argmax(counts)

        # best probability is just a proportion of a major class
        probability = max(counts[0] / (counts[0] + counts[1]), counts[1] / (counts[0] + counts[1]))

        return [best_class, probability]

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
            Calculate derivative of Binary Cross Entropy Loss with respect to y_pred vector

            Input:
                y_true: correct (target) values
                y_pred: predicted values
            Output:
                vector of derivatives with respect to each y_pred
        """

        return -(y_true - y_pred)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
            Calculate the second derivative (hessian) for the Binary Cross Entropy

            Input:
                y_true: correct (target) values
                y_pred: predicted values
            Output:
                vector of second derivatives with respect to each y_pred

        """

        return y_pred * (1 - y_pred)
