import numpy as np
from mlzero.metrics.distance_functions import manhattan_distance, euclidean_distance
from scipy.stats import mode

class KNeighborsClassifier:

    """
        K Nearest Neighbors classifier
    """

    def __init__(self, n_neighbors: int = 5, distance_function: str = 'l2'):
        """
            Input:
                n_neighbors: number of nearest neighbors to make a prediction
                distance_function: define how to calculate distance between 2 vectors
                                   'l1' - manhattan distance (sum of absolute values)
                                   'l2' - euclidean distance (sum of squared residuals)
            Output:
                -
        """
        self.n_neighbors = n_neighbors

        if(distance_function.lower() == 'l1'):
            self.distance_function = manhattan_distance
        elif(distance_function.lower() == 'l2'):
            self.distance_function = euclidean_distance
        else:
            raise ValueError(f"Unknown distance function {distance_function}")


    def fit(self, X: np.ndarray, y: np.ndarray):
        """

            Input:
                X: input vector with shape (n_samples, n_features)
                y: target values for each input vector with shape (n_samples, )
            Output
                self object
        """

        # save data for the future predictions
        self.X_ = X.copy()
        self.y_ = y.copy()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Input:
                X: input vectors with shape (n_samples, n_features)
            Output:
                predicted labels for each sample with shape (n_samples,)
        """

        y_predictions = []

        for x in X: # iterate over each sample

            # get distances to all other vectors from the current vector
            distances = [self.distance_function(x, x_) for x_ in self.X_]

            # get idx of sorted by distance vectors
            idx = np.argsort(distances)

            # get K nearest vector's labels
            top_k_labels = self.y_[idx[:self.n_neighbors]]

            # add most frequent label in nearest vectors
            y_predictions.append(mode(top_k_labels, axis = None)[0][0])

        return np.array(y_predictions)
