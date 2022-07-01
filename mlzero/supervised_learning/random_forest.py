import numpy as np
from mlzero.utils.data import bootstrap, get_most_frequent_value
from mlzero.supervised_learning.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestClassifier:

    """
        The Random Forest Classifier is an ensemble which builds a few different classification trees using
        boostrap idea for data preparation and then average their predictions.

        Basically there are 3 main steps:

        1. Build decision trees:
            For each tree:
                1.1 Get sub-sample of the original data using boostrap (randomly picking elements with repeats)
                1.2 Train a classification tree but for each split we consider only "max_features" randomly picked
                    features from the all input features

        2. Usage of Random Foreste
            Here we just use classification by most votes. For example if we have 6 trees
            and 5 predicted "Yes" and 1 predicted "No" -> final prediction is "Yes"

        3. (This step is not necessary) Estimation of Random Forest with Out-Of-Bag. If we use Bootstrapping for
            generating sub-samples on each step we have some "extra" input entries which are not used during building a
            tree. So we can collect all such samples during all steps (tree buildings) and such dataset will be called
            as "Out-Of-Bag dataset". Which we can use for RandomForest quality estimation.

            Note:
                1. We get "Out-Of-Bag dataset" at each step when we are bootstrapping data for a tree -> after N steps we
                   have N "Out-Of-Bag dataset" -> we combine it all together and have final, big "Out-Of-Bag dataset"
                   which we use for OOB score for a Random Forest
                2. Out-Of-Baf dataset can have repeats because we take entries on different steps independently

        Links:
            1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html [EN]
            2. https://www.youtube.com/watch?v=J4Wdy0Wc_xQ [EN]
    """

    def __init__(self, n_estimators: int, bootstrap: bool = True, tree_parameters: dict = None):
        """

            Input:
                n_estimators: how many trees we want to have in a forest
                bootstrap: use boostrap sampling for building trees or not
                           False = the whole dataset is used to build each tree
                tree_parameters: each tree will have these parameters
            Output:

        """

        self.n_estimators = n_estimators
        self.boostrap = bootstrap
        self.tree_parameters = tree_parameters
        self.trees = []


        if (self.tree_parameters is None):
            self.tree_parameters = {}

        if ('max_features' not in self.tree_parameters):
            self.tree_parameters['max_features'] = 'sqrt'



    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Build Random Forest, e.g train each of n_estimators tree

            Input:
                X: 2d array, where 1 dimension is an object and 2 dimension is object features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted RandomForestClassifier object
        """

        for i in range(self.n_estimators):

            if(self.boostrap):
                bootstrapped_X, boostrapped_y = bootstrap(X, y)
            else:
                bootstrapped_X, boostrapped_y = X, y

            self.trees.append(DecisionTreeClassifier(**self.tree_parameters).fit(bootstrapped_X, boostrapped_y))

        return self

    def predict(self, X: np.ndarray):
        """
            Make a prediction for the input array

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predictions for each input sample
        """

        predictions_ = []

        for i in range(self.n_estimators):
            predictions_.append(self.trees[i].predict(X))

        # convert list to numpy array
        predictions_ = np.array(predictions_).T

        result = []

        for prediction in predictions_:
            result.append(get_most_frequent_value(prediction))

        return np.array(result)


class RandomForestRegressor:

    """
        The Random Forest Regressor is an ensemble which builds a few different regression trees using
        boostrap idea for data preparation and then average their predictions.

        Basically there are 3 main steps:

        1. Build decision trees:
            For each tree:
                1.1 Get sub-sample of the original data using boostrap (randomly picking elements with repeats)
                1.2 Train a regression tree but for each split we consider only "max_features" randomly picked
                    features from the all input features

        2. Usage of Random Forest

            Here we just use mean of all predictions

        3. (This step is not necessary) Estimation of Random Forest with Out-Of-Bag. If we use Bootstrapping for
            generating sub-samples on each step we have some "extra" input entries which are not used during building a
            tree. So we can collect all such samples during all steps (tree buildings) and such dataset will be called
            as "Out-Of-Bag dataset". Which we can use for RandomForest quality estimation.

            Note:
                1. We get "Out-Of-Bag dataset" at each step when we are bootstrapping data for a tree -> after N steps we
                   have N "Out-Of-Bag dataset" -> we combine it all together and have final, big "Out-Of-Bag dataset"
                   which we use for OOB score for a Random Forest
                2. Out-Of-Baf dataset can have repeats because we take entries on different steps independently

        Links:
            1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html [EN]
            2. https://www.youtube.com/watch?v=J4Wdy0Wc_xQ [EN]
    """

    def __init__(self, n_estimators: int, bootstrap: bool = True, tree_parameters: dict = None):
        """

            Input:
                n_estimators: how many trees we want to have in a forest
                bootstrap: use boostrap sampling for building trees or not
                           False = the whole dataset is used to build each tree
                tree_parameters: each tree will have these parameters
            Output:

        """

        self.n_estimators = n_estimators
        self.boostrap = bootstrap
        self.tree_parameters = tree_parameters
        self.trees = []


        if (self.tree_parameters is None):
            self.tree_parameters = {}

        if ('max_features' not in self.tree_parameters):
            self.tree_parameters['max_features'] = 'sqrt'



    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Build Random Forest, e.g train each of n_estimators tree

            Input:
                X: 2d array, where 1 dimension is an object and 2 dimension is object features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted RandomForestRegressor object
        """

        for i in range(self.n_estimators):

            if(self.boostrap):
                bootstrapped_X, boostrapped_y = bootstrap(X, y)
            else:
                bootstrapped_X, boostrapped_y = X, y

            self.trees.append(DecisionTreeRegressor(**self.tree_parameters).fit(bootstrapped_X, boostrapped_y))

        return self

    def predict(self, X: np.ndarray):
        """
            Make a prediction for the input array

            Input:
                X: 2d array, where 1 dimension is an object and 2 dimension is a object features
            Output:
                1d array of predictions for each input sample
        """

        predictions_ = []

        # collect predictions from all trees
        for i in range(self.n_estimators):
            predictions_.append(self.trees[i].predict(X))

        # convert list to numpy array and transpose
        # row now contains different predictions for one sample
        predictions_ = np.array(predictions_).T

        result = []

        #
        for prediction in predictions_:
            result.append(np.mean(prediction))

        return np.array(result)
