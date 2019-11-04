import numpy as np

class SquareLoss:
    def __call__(self, y_pred, y):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y_pred, y):
        return -(y - y_pred)

class MeanSquaredError:
    def __call__ (self, predict_labels, true_labels):

        #check that we have vectors with equal size
        assert(len(predict_labels) == len(true_labels))

        n = len(predict_labels)
        # check that size of vectors is not null
        assert (n > 0)

        #calculate sum of squares of difference between predicted and original labels
        sum = np.sum(np.power(true_labels - predict_labels, 2))

        return sum / n

    def gradient(self, predict_labels, true_labels):
        #check that we have vectors with equal size
        assert(len(predict_labels) == len(true_labels))

        n = len(predict_labels)
        #check that size of vectors is not null
        assert (n > 0)

        return -2.0 * np.subtract(true_labels, predict_labels) / n





ans = MeanSquaredError()

#print(ans(np.array([1, 2]), np.array([2, 2])))
#print(ans.gradient(np.array([1, 2]), np.array([2, 2])))