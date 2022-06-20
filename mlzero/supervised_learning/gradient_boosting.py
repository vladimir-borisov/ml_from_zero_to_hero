from mlzero.metrics.loss_functions import MeanSquaredError, SquaredError, BinaryCrossEntropy
from mlzero.metrics.activation_functions import Sigmoid
from mlzero.supervised_learning.decision_tree import BoostingDecisionTreeRegressor
import numpy as np


#TODO:
# 1. For regression and classification add support of any loss function for that:
#   1.1 Add loss_fn variable
#   1.2 For classification add activation function specific for each loss
#   1.3 Add loss minimization with respect to loss function in estimators (for leaves in regression trees)


class GradientBoostingRegressor:
    """
        Gradient Boosting for Regression is an ensemble method which uses base estimators for prediction of
        gradient of loss function. The main idea is that we use sum of many base learners to make
        our prediction precise. We start we initial prediction and after that train estimator
        to correct this initial prediction.

        It's called GRADIENT boosting because we use gradient descent idea for a loss minimization.

        Notes:
            1. In gradient boosting we always optimize some loss function with base estimators
            2. The initial prediction calculated with respect to minimum of loss function it's not a random point
            3. If you use a regression tree as a base learner -> you also optimize loss function in each leaf
               it's vey important to remember people always just predict -gradient and don't worry about optimization
               in leaves



        Links:
            1. Original paper: https://statweb.stanford.edu/~jhf/ftp/trebst.pdf [EN]
            2. Scikit-Learn: https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation [EN]
            3. StatQuest. Josh Starmer: youtube.com/watch?v=3CC4N4z3GJc [EN]
            4. ODS article: https://habr.com/ru/company/ods/blog/327250/ [RU]
    """

    def __init__(self, loss: str = 'squared_error', learning_rate: float = 0.1, n_estimators: int = 100,
                 estimator_type: str = 'decision_tree', estimator_parameters: dict = None):
        """

            Input:
                loss: string value which loss for learners evaluation
                learning_rate: how strong each learner change prediction
                n_estimators: how many learners in the ensemble
                estimator_type: which type of learners (models) is using in the ensemble
                estimator_parameters: every estimator will be initialized with this parameters
            Output:
                GradientBoostingRegressor object
        """

        self.loss_name = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimator_type = estimator_type
        self.estimator_parameters = estimator_parameters
        self.estimators_ = []

        # we didn't get parameters let's set default
        if (self.estimator_parameters is None):
            self.estimator_parameters = {}
            self.estimators_parameters['loss'] = self.loss_name

        if ( self.loss_name == 'squared_error'):
            self.loss_function = SquaredError()
        elif (self.loss_name == 'mse'):
            self.loss_function = MeanSquaredError()


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Find an optimal parameter for the given input (X) and output (y) values.
            First, we predict a mean along all output values and save it as a self.base_value
            After that, we train "self.n_estimators" models to correct base value to the right side

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted object of GradientBoostingRegressor
        """

        self.base_prediction = self.loss_function.optimal_value(y_true=y)

        # set default prediction as mean among all target values
        y_pred = np.full(shape = len(y), fill_value = self.base_prediction)

        for i in range(self.n_estimators):

            gradients = self.loss_function.gradient(y_true = y, y_pred = y_pred)

            estimator = BoostingDecisionTreeRegressor(**self.estimator_parameters).fit(X, y_true=-gradients, y_prev=y_pred)

            self.estimators_.append(estimator)

            # recalculate prediction with the current estimator
            y_pred += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Make prediction for the given input values

            For each input we take a base value and after that trying to make it better with fitted estimators.
            Each estimator's answer is multiplied by the same learning rate.

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predicted labels
        """

        y_pred = np.full(shape = len(X), fill_value = self.base_prediction)

        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)

        return y_pred


class GradientBoostingClassifier:
    """
        Gradient Boosting for Classification is an ensemble method which uses base estimators for prediction of
        gradient of loss function. The main idea is that we use sum of many base learners to make
        our prediction precise. We start with an initial prediction and after that train estimator
        to correct this initial prediction.

        It's called GRADIENT boosting because we use gradient descent idea for a loss minimization.

        Notes:
            1. For classification problem we still use regression models as base estimators!
            2. In gradient boosting we always optimize some loss function with base estimators
            3. The initial prediction calculated with respect to minimum of loss function it's not a random point
            4. If you use a regression tree as a base learner -> you also optimize loss function in each leaf
               it's vey important to remember people always just predict -gradient and don't worry about optimization
               in leaves

        Links:
            1. Original paper: https://statweb.stanford.edu/~jhf/ftp/trebst.pdf [EN] (page 1198, section 4.5)
            2. Scikit-Learn: https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation [EN]
            3. StatQuest. Josh Starmer: https://www.youtube.com/watch?v=jxuNLH5dXCs [EN]
            4. ODS article: https://habr.com/ru/company/ods/blog/327250/ [RU]
    """

    def __init__(self, loss: str = 'log_loss', learning_rate: float = 0.1, n_estimators: int = 100,
                 estimator_type: str = 'decision_tree', estimator_parameters: dict = None):
        """

            Input:
                loss: string value which loss for learners evaluation
                learning_rate: how strong each learner change prediction
                n_estimators: how many learners in the ensemble
                estimator_type: which type of learners (models) is using in the ensemble
                estimator_parameters: every estimator will be initialized with this parameters
            Output:
                GradientBoostingRegressor object
        """

        self.loss_name = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimator_type = estimator_type
        self.estimator_parameters = estimator_parameters
        self.estimators_ = []

        if (self.estimator_parameters is None):
            self.estimator_parameters = {}

        if (self.loss_name == 'log_loss'):
            self.loss_function = BinaryCrossEntropy()
            self.activation_function = Sigmoid()  # this function helps us to convert log(odds) -> probability

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Find an optimal parameter for the given input (X) and output (y) values.
            First, we predict a mean along all output values and save it as a self.base_value
            After that, we train "self.n_estimators" models to correct base value to the right side

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
                y: 1d array, target value for corresponding features in X
            Output:
                fitted object of GradientBoostingRegressor


            Note:
                base estimators actually don't predict PROBABILITIES they predict some real value
                usually LOG(ODDS)
        """


        self.base_prediction = self.activation_function.inverse(self.loss_function.optimal_value(y)[1]) # first we get optimal log(odds) value

        # but it's usually a probability for a classification -> we need to convert it to

        # set default prediction as mean among all target values
        y_pred = np.full(shape = len(y), fill_value = self.base_prediction)


        for i in range(self.n_estimators):

            gradients = self.loss_function.gradient(y_true=y, y_pred=y_pred)

            # note that we don't use a gradient of l
            estimator = BoostingDecisionTreeRegressor(**self.estimator_parameters).fit(X, y_true=-gradients, y_prev=self.activation_function(y_pred))

            self.estimators_.append(estimator)

            # recalculate prediction with the current estimator
            y_pred += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Make prediction for the given input values

            For each input we take a base value and after that trying to make it better with fitted estimators.
            Each estimator's answer is multiplied by the same learning rate.

            Input:
                X: 2d array, where 1 dimension is a number of samples and 2 dimension is a features
            Output:
                1d array of predicted probabilities
        """

        y_pred = np.full(shape = len(X), fill_value = self.base_prediction)

        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)

        return self.activation_function(y_pred)
