import numpy as np
from typing import Optional
from mlzero.metrics.loss_functions import MeanSquaredError, BinaryCrossEntropy, SquaredError

# TODO: move to metrics folder
def gini_index(x: np.ndarray) -> float:
    """
        Gini Index is a measure of impurity in a distribution
        Impurity means how many different values are in the distribution

        0 - perfect classification (only one value)
        1 - bad classification (too many different values)

        Input:
            x: 1d array, distribution of values
        Output:
            gini index - number in range [0, 1]
    """

    unique_values = np.unique(x, return_counts = True)  # get all unique values from y

    gini_index = 1

    for value, count in zip(unique_values[0], unique_values[1]):
        normalized_count = count / len(x)
        gini_index -= normalized_count * normalized_count

    return gini_index

class DecisionTreeNode:
    """
        This class implements a base element of a decision tree - tree node
        In a node we keep the information which help us to make a decision :)

        For example, it can be information about a rule for a data split or a target class for a leaf node
    """

    def __init__(self, value: Optional[float] = None, feature_index: Optional[int] = None, split_value: Optional[float] = None,
                 left_child_node = None, right_child_node = None):
        """

            Input:
                value: only for a leaf node, this value we return as predicted values
                feature_index: index of a feature in the input array. this features we use for splitting a node
                threshold: threshold values for splitting a node. used for non a leaf node
                left_child_node: this node contains all sample which follow splitting criteria
                right_child_node: this node contains all samples which doesn't follow splitting criteria
            Output:

        """
        self.value = value
        self.feature_index = feature_index
        self.threshold = split_value
        self.left_child_node = left_child_node # this node like "True" node -> feature value >= threshold
        self.right_child_node = right_child_node # this node like "False" node -> feature value < threshold

    @property
    def is_leaf(self) -> bool:
        """
            Is this node a leaf (node without children) node?
        """
        return self.value is not None

    def get_child_node(self, X: np.ndarray):
        """
            Get a child node using current node + input features + threshold

            Input:
                X: 1d array of the input features in order as it was during training
            Output:
                left or right child node
        """

        if (X[self.feature_index] >= self.threshold):
            return self.left_child_node
        else:
            return self.right_child_node


# TODO: add categorical features
class DecisionTreeClassifier:

    """
        Decision tree is an algorithm which is based on a series of questions
        which helps to split data with different output labels
        In a code it looks like a graph data structure, usually binary tree but sometimes not

        There are 4 most known algorithms for building decision tree:

        1. ID3 - not necessary a binary tree, work only with categorical features, splitting criteria is information gain(IG)
                 only classification task can be solved by this algo
                 (Quinlan, J. R. 1986. Induction of Decision Trees) [EN]

        2. C4.5 - modification of ID3 with support of continuous features + handling missing data +
                  allow features to have different costs, splitting criteria is normalized information gain
                  also added pruning (simplification of a tree via replacement not important branches with leafs)
                  but still only classification task can be solved by this algo
                  (Quinlan, J. R. C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers) [EN]

        3. C5.0 - modification of C4.5, speed and memory optimization are added, support of boosting for trees and
                  also using "winnow" algorithm to remove unuseful features
                  and still no regression output here :)

                  winnow algo (https://www.cs.yale.edu/homes/aspnes/pinewiki/WinnowAlgorithm.html[EN])
                  C4.5 vs C5.0 (https://www.rulequest.com/see5-comparison.html [EN])

        4. CART - very similar to C4.5 but can be applied to a regression task (i.e predict continuous variable like
                  salary, temperature etc.), splitting criteria is gini index

        This realization doesn't follow any of the algorithms above. It's very straightforward without any optimization
        like pruning, memory or speed. Simplicity is the main goal :)

        It also doesn't support categorical features.

        Links:
            1. https://scikit-learn.org/stable/modules/tree.html [EN]
    """


    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None,
                 min_samples_split: int = 2, max_features = None, loss = 'log_loss'):
        """

            Input:
                criterion: which criteria we use for splitting a node
                max_depth: max depth of a tree. depth - number of nodes from root to a particular node
                min_samples_split: min number of values in array for splitting a node
                max_features: each split we will check randomly not more than "max_features" even if we have more
                loss: loss function which we optimize by tree in leaves
            Output:


            Note:
                1. value in leaves depends on the loss function we try to minimize

        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_root = None

        # binary cross entropy loss or log loss
        if (loss == 'log_loss'):
            self.loss = BinaryCrossEntropy()


    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """
            Calculate value in a leaf

            Input:
                y: target values in a lead
            Output:
                one value - prediction for this leaf


            Notes:
                1. it's especially important when we use tree in boosting because in each leaf we need to minimize
                   loss function -> we have to be able to change calculation in leaves
        """

        y_unique, counts = np.unique(y, return_counts=True)
        ind = np.argmax(counts)

        return y_unique[ind]

    def __build_tree__(self, X: np.ndarray, y: np.ndarray, depth_level: int = 1):
        """
            Inner function which creates a binary tree

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
                depth_level: number of nodes from root node to the current node
            Output:
                DecisionTree root node
        """

        n_samples, n_features = X.shape

        X_transposed = X.T

        """ First as it's a recursion we define a terminal conditions for stopping infinity recursion """
        # check if this node is a leaf node
        if (gini_index(y) == 1):  # if all labels are equal lets predict this label
            return DecisionTreeNode(value=self._calculate_leaf_value(y))


        # if one of the conditions is met -> set this node as a leaf and return the most frequent value
        if (((self.max_depth is not None) and (self.max_depth <= depth_level)) or (self.min_samples_split >= n_samples)):
            return DecisionTreeNode(value=self._calculate_leaf_value(y))

        best_split_params = {
            'left_node_indices': None,
            'right_node_indices': None,
            'gini_index_left': 1,
            'gini_index_right': 1,
            'feature_idx': -1,
            'split_value': None
        }

        # let's find the best feature for splitting

        features_random_ids = self.max_features * [1] + (n_features - self.max_features) * [0] # choose random feature ids
        np.random.shuffle(features_random_ids)


        for feature_idx, features_status in enumerate(features_random_ids):

            # 0 means we don't consider this feature
            if (features_status == 0):
                continue

            features = X_transposed[feature_idx]

            unique_features = np.unique(features)

            # it doesn't make sense divide array if we have only one value
            # because all elements will be in the same node
            if (len(unique_features) == 1):
                continue

            for unique_feature in unique_features:

                # find all the samples which
                left_node_indices = features >= unique_feature
                right_node_indices = features < unique_feature

                # divide X and y for the left and right branches
                # there is no copy here so don't change arrays or deepcopy them
                X_left = X[left_node_indices, :]
                y_left = y[left_node_indices]

                X_right = X[right_node_indices, :]
                y_right = y[right_node_indices]

                gini_index_left = gini_index(y_left)
                gini_index_right = gini_index(y_right)

                # check if this split is making out gini index better
                # we just sum a gini coefficient in child nodes
                if (gini_index_left + gini_index_right <
                    best_split_params['gini_index_left'] + best_split_params['gini_index_right']):

                    best_split_params['left_node_indices'] = left_node_indices
                    best_split_params['right_node_indices'] = right_node_indices
                    best_split_params['gini_index_left'] = gini_index_left
                    best_split_params['gini_index_right'] = gini_index_right
                    best_split_params['feature_idx'] = feature_idx
                    best_split_params['split_value'] = unique_feature

        # we couldn't find a features for divide so let's make it a leaf
        if (best_split_params['feature_idx'] == -1):
            return DecisionTreeNode(value=self._calculate_leaf_value(y))

        X_left = X[best_split_params['left_node_indices'], :]
        y_left = y[best_split_params['left_node_indices']]

        X_right = X[best_split_params['right_node_indices'], :]
        y_right = y[best_split_params['right_node_indices']]

        # we found best split
        left_node = self.__build_tree__(X_left, y_left, depth_level + 1)
        right_node = self.__build_tree__(X_right, y_right, depth_level + 1)

        return DecisionTreeNode(feature_index= best_split_params['feature_idx'],
                                left_child_node = left_node, right_child_node = right_node,
                                split_value = best_split_params['split_value'])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Create decision tree from input features X and their labels y

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted object of DecisionTree class
        """

        n_samples, n_features = X.shape

        if (self.max_features is None):
            self.max_features = n_features
        elif (isinstance(self.max_features, str)):
            if (self.max_features == 'sqrt'):
                self.max_features = int(np.sqrt(n_features))


        self.tree_root = self.__build_tree__(X, y)

        return self

    def predict(self, X: np.ndarray):
        """
            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predicted labels
        """

        if (self.tree_root is None):
            raise ValueError("The Decision Tree is not fitted yet")

        y_ = []

        for x in X:

            # we start from root node
            head = self.tree_root

            # go deep until we reach a leaf
            # it takes O(max deep) iterations in the worst case
            while(not head.is_leaf):
                head = head.get_child_node(x) # go down to a child node

            # now we know that head is a leaf node so let's return a prediction

            y_.append(head.value)

        return np.array(y_)

class DecisionTreeRegressor:

    """
        Decision Tree for Regression is the same as for classification but splitting criteria is different
        Here we use MSE as a splitting criteria

        Main idea is to pick a feature for splitting, then try to split and calculate MSE,
        if MSE is ok -> save the split and continue building a tree

        Links:
            1. https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047 [EN]
            2. Splitting criteria. https://stats.stackexchange.com/questions/220350/regression-trees-how-are-splits-decided [EN]
    """


    def __init__(self, criterion: str = "mse", max_depth: Optional[int] = None,
                 min_samples_split: int = 2, max_features = None):
        """

            Input:
                criterion: which criteria we use for splitting a node
                max_depth: max depth of a tree. depth - number of nodes from root to a particular node
                min_samples_split: min number of values in array for splitting a node
                max_features: each split we will check randomly not more than "max_features" even if we have more
                              if None consider all features
            Output:

        """
        self.criterion = criterion

        if (criterion == 'mse'):
            self.loss_function = MeanSquaredError()


        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_root = None

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """
            Calculate value in a leaf

            Note: it's especially important when we use tree in boosting because in each leaf we need to minimize
                  loss function -> we have to be able to change calculation in leaves

            Input:
                y: target values in a lead
            Output:
                one value - prediction for this leaf
        """

        return self.loss_function.optimal_value(y)

    def __build_tree(self, X: np.ndarray, y: np.ndarray, depth_level: int = 0):
        """
            Inner function which creates a binary tree

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
                depth_level: number of nodes from root node to the current node
            Output:
                DecisionTree root node
        """

        n_samples, n_features = X.shape

        X_transposed = X.T

        """ First as it's a recursion we define a terminal conditions for stopping infinity recursion """
        # check if this node is a leaf node
        if (len(np.unique(y)) == 1):  # if all labels are equal lets predict this label
            return DecisionTreeNode(value=self._calculate_leaf_value(y))


        # if one of the conditions is met -> set this node as a leaf and return the most frequent value
        if (((self.max_depth is not None) and (self.max_depth <= depth_level)) or (self.min_samples_split >= n_samples)):
            return DecisionTreeNode(value=self._calculate_leaf_value(y))

        best_split_params = {
            'left_node_indices': None,
            'right_node_indices': None,
            'mse_left': 100000000,
            'mse_right': 100000000,
            'feature_idx': -1,
            'split_value': None
        }

        # let's find the best feature for splitting

        features_random_ids = self.max_features * [1] + (n_features - self.max_features) * [0]  # choose random feature ids
        np.random.shuffle(features_random_ids)

        for feature_idx, features_status in enumerate(features_random_ids):

            # 0 means we don't consider this feature
            if (features_status == 0):
                continue

            features = X_transposed[feature_idx]

            unique_features = np.unique(features)

            # it doesn't make sense divide array if we have only one value
            # because all elements will be in the same node
            if (len(unique_features) == 1):
                continue

            for unique_feature in unique_features:

                # find all the samples which
                left_node_indices = features >= unique_feature
                right_node_indices = features < unique_feature

                # divide X and y for the left and right branches
                # there is no copy here so don't change arrays or deepcopy them
                X_left = X[left_node_indices, :]
                y_left = y[left_node_indices]

                X_right = X[right_node_indices, :]
                y_right = y[right_node_indices]

                # we can't split this node by this feature value
                if (len(y_left) == 0 or len(y_right) == 0):
                    continue

                mse_left = self.loss_function(y_left, np.full(len(y_left), np.mean(y_left)))
                mse_right = self.loss_function(y_right, np.full(len(y_right), np.mean(y_right)))

                # check if this split is making out gini index better
                # we just sum a gini coefficient in child nodes
                # TODO: Think about splitting criteria because current is bad for example for sin(x)
                if (mse_left + mse_right <
                    best_split_params['mse_left'] + best_split_params['mse_right']):

                    best_split_params['left_node_indices'] = left_node_indices
                    best_split_params['right_node_indices'] = right_node_indices
                    best_split_params['mse_left'] = mse_left
                    best_split_params['mse_right'] = mse_right
                    best_split_params['feature_idx'] = feature_idx
                    best_split_params['split_value'] = unique_feature

        # we couldn't find a features for divide so let's make it a leaf
        if (best_split_params['feature_idx'] == -1):
            return DecisionTreeNode(value=self._calculate_leaf_value(y))

        X_left = X[best_split_params['left_node_indices'], :]
        y_left = y[best_split_params['left_node_indices']]

        X_right = X[best_split_params['right_node_indices'], :]
        y_right = y[best_split_params['right_node_indices']]

        # we found best split
        left_node = self.__build_tree(X_left, y_left, depth_level + 1)
        right_node = self.__build_tree(X_right, y_right, depth_level + 1)

        return DecisionTreeNode(feature_index= best_split_params['feature_idx'],
                                left_child_node = left_node, right_child_node = right_node,
                                split_value = best_split_params['split_value'])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Create decision tree from input features X and their labels y

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted object of DecisionTreeRegressor class
        """

        n_samples, n_features = X.shape

        if (self.max_features is None):
            self.max_features = n_features
        elif (isinstance(self.max_features, str)):
            if (self.max_features == 'sqrt'):
                self.max_features = int(np.sqrt(n_features))

        self.tree_root = self.__build_tree(X, y)

        return self

    def predict(self, X: np.ndarray):
        """
            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predicted labels
        """

        if (self.tree_root is None):
            raise ValueError("The Decision Tree is not fitted yet")

        y_ = []

        for x in X:

            # we start from root node
            head = self.tree_root

            # go deep until we reach a leaf
            # it takes O(max deep) iterations in the worst case
            while(not head.is_leaf):
                head = head.get_child_node(x) # go down to a child node

            # now we know that head is a leaf node so let's return a prediction

            y_.append(head.value)

        return np.array(y_)

class BoostingDecisionTreeRegressor:
    """
        This is just a modification of DecisionTreeRegressor for Gradient Boosting Classification

        It's almost the same, but we update values in leaves differently because we use additive function
        and use values from previous prediction in case of boosting

        Simple Tree: min(loss(y_true, y_pred))
        Boosted Tree: min(loss(y_true, y_prev + y_pred))

        In both cases we are trying to minimize a loss function changing y_pred but
        for boosting we also consider values from the previous steps (trees)

        Links:
            1. https://www.youtube.com/watch?v=StWY5QWMXCw [EN] (Check step 2, point C)
    """


    def __init__(self, criterion: str = "mse", max_depth: Optional[int] = None,
                 min_samples_split: int = 2, max_features = None, loss = 'log_loss'):
        """

            Input:
                criterion: which criteria we use for splitting a node
                max_depth: max depth of a tree. depth - number of nodes from root to a particular node
                min_samples_split: min number of values in array for splitting a node
                max_features: each split we will check randomly not more than "max_features" even if we have more
                              if None consider all features
                loss: loss function which we optimize by tree in leaves
                      log_loss - for classification
                      squared error - for regression
            Output:


            Note:
                1. value in leaves depends on the loss function we try to minimize

        """

        self.loss_name = loss
        self.criterion_name = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_root = None

        # this function is used for splittiong a node
        if (self.criterion_name == 'mse'):
            self.criterion_function = MeanSquaredError()
        if (self.criterion_function == 'squared_error'):
            self.criterion_function = SquaredError()


    def _calculate_leaf_value(self, y_true: np.ndarray, y_prev: np.ndarray) -> float:
        """
            Calculate the optimal value in a leaf

            Input:
                y_true: target values in a leaf (in this case it's called "Residuals")
                y_prev: values from a previous step (probabilities not log(odds))!!!
            Output:
                one value (log(odds)) - optimal value for this leaf with respect to loss function

            Note:
                1. the output value depends on a loss function
        """

        # for this case we use a second order Taylor Polynomial
        # this is true only for log_loss !!!
        if (self.loss_name == 'log_loss'):
            return np.sum(y_true) / np.sum(y_prev * (1 - y_prev))

        # note that for squared error it does not depends on the previous values
        if (self.loss_name == 'squared_error'):
            return np.mean(y_true)

        #TODO: should it be solve for multiclass or just using N binary classifiers is ok?

        raise ValueError(f"Can't calculate a leaf value for the {self.loss_name}")

    def __build_tree__(self, X: np.ndarray, y_true: np.ndarray, y_prev: np.ndarray, depth_level: int = 1):
        """
            Inner function which creates a binary tree

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y_true: 1d array, target values in a leaf (in this case it's called "Residuals")
                y_prev: 1d array, predicted values from the previous step (probabilities not log(odds))!!!
                depth_level: number of nodes from root node to the current node
            Output:
                DecisionTree root node
        """

        n_samples, n_features = X.shape

        X_transposed = X.T

        """ First as it's a recursion we define a terminal conditions for stopping infinity recursion """
        # check if this node is a leaf node
        if (gini_index(y_true) == 1):  # if all labels are equal lets predict this label
            return DecisionTreeNode(value=self._calculate_leaf_value(y_true = y_true, y_prev = y_prev))


        # if one of the conditions is met -> set this node as a leaf and return the most frequent value
        if (((self.max_depth is not None) and (self.max_depth <= depth_level)) or (self.min_samples_split >= n_samples)):
            return DecisionTreeNode(value=self._calculate_leaf_value(y_true = y_true, y_prev = y_prev))

        best_split_params = {
            'left_node_indices': None,
            'right_node_indices': None,
            'criterion_left': 100000000,
            'criterion_right': 100000000,
            'feature_idx': -1,
            'split_value': None
        }

        # let's find the best feature for splitting

        features_random_ids = self.max_features * [1] + (n_features - self.max_features) * [0]  # choose random feature ids
        np.random.shuffle(features_random_ids)

        for feature_idx, features_status in enumerate(features_random_ids):

            # 0 means we don't consider this feature
            if (features_status == 0):
                continue

            features = X_transposed[feature_idx]

            unique_features = np.unique(features)

            # it doesn't make sense divide array if we have only one value
            # because all elements will be in the same node
            if (len(unique_features) == 1):
                continue

            for unique_feature in unique_features:

                # find all the samples which
                left_node_indices = features >= unique_feature
                right_node_indices = features < unique_feature

                # divide X and y for the left and right branches
                # there is no copy here so don't change arrays or deepcopy them
                y_true_left = y_true[left_node_indices]
                y_true_right = y_true[right_node_indices]

                # we can't split this node by this feature value
                if (len(y_true_left) == 0 or len(y_true_right) == 0):
                    continue

                criterion_left = self.criterion_function(y_true_left, np.full(len(y_true_left), self.criterion_function.optimal_value(y_true_left)))
                criterion_right = self.criterion_function(y_true_right, np.full(len(y_true_right), self.criterion_function.optimal_value(y_true_left)))

                # TODO: Think about splitting criteria because current is bad for example for sin(x)
                if (criterion_left + criterion_right <
                    best_split_params['criterion_left'] + best_split_params['criterion_right']):

                    best_split_params['left_node_indices'] = left_node_indices
                    best_split_params['right_node_indices'] = right_node_indices
                    best_split_params['criterion_left'] = criterion_left
                    best_split_params['criterion_right'] = criterion_right
                    best_split_params['feature_idx'] = feature_idx
                    best_split_params['split_value'] = unique_feature

        # we couldn't find a features for divide so let's make it a leaf
        if (best_split_params['feature_idx'] == -1):
            return DecisionTreeNode(value=self._calculate_leaf_value(y_true = y_true, y_prev = y_prev))


        # here we get deep copy not pointers from original array

        X_left = X[best_split_params['left_node_indices'], :]
        y_true_left = y_true[best_split_params['left_node_indices']]
        y_prev_left = y_prev[best_split_params['left_node_indices']]


        X_right = X[best_split_params['right_node_indices'], :]
        y_true_right = y_true[best_split_params['right_node_indices']]
        y_prev_right = y_prev[best_split_params['right_node_indices']]

        # we found best split
        left_node = self.__build_tree__(X_left, y_true_left, y_prev_left, depth_level + 1)
        right_node = self.__build_tree__(X_right, y_true_right, y_prev_right, depth_level + 1)

        return DecisionTreeNode(feature_index= best_split_params['feature_idx'],
                                left_child_node = left_node, right_child_node = right_node,
                                split_value = best_split_params['split_value'])

    def fit(self, X: np.ndarray, y_true: np.ndarray, y_prev: np.ndarray):
        """
            Create decision tree from input features X and their labels y

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y_true: 1d array, target value for corresponding features in X
                y_prev: 1d array, predicted values from the previous step ((probabilities not log(odds))!!!)
            Output:
                fitted object of DecisionTree class
        """

        n_samples, n_features = X.shape

        if (self.max_features is None):
            self.max_features = n_features
        elif (isinstance(self.max_features, str)):
            if (self.max_features == 'sqrt'):
                self.max_features = int(np.sqrt(n_features))

        self.tree_root = self.__build_tree__(X, y_true = y_true, y_prev = y_prev)

        return self

    def predict(self, X: np.ndarray):
        """
            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predicted labels
        """

        if (self.tree_root is None):
            raise ValueError("The Decision Tree is not fitted yet")

        y_ = []

        for x in X:

            # we start from a root node
            head = self.tree_root

            # go deep until we reach a leaf
            # it takes O(max deep) iterations in the worst case
            while(not head.is_leaf):
                head = head.get_child_node(x)  # go down to a child node

            # now we know that head is a leaf node so let's return a prediction

            y_.append(head.value)

        return np.array(y_)