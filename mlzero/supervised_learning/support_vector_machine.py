import numpy as np
from typing import Optional, Callable
from mlzero.metrics.kernel_functions import *
from mlzero.metrics.loss_functions import HingeLoss
import cvxopt

#TODO: write function descriptions

# Turn off CVXOPT output
cvxopt.solvers.options['show_progress'] = False

class SupportVectorClassifier:
    """
        Support Vector Classifier based on QP(Quadratic Problem) optimization with CVXOPT library

        Naming of parameters is very close to Scikit-Learn

        Links:

            1. https://blog.dominodatalab.com/fitting-support-vector-machines-quadratic-programming [ENG] (QP optimization)
            2. https://neerc.ifmo.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D1%85_%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2_(SVM) [RU] (QP optimization)
            3. https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8 [ENG] (QP optimization)
            4. https://xavierbourretsicotte.github.io/SVM_implementation.html [ENG] (QP using CVXOPT)
    """


    def __init__(self, C: float = 1.0, kernel: str = 'linear',
                 degree: int = 3, gamma: Optional[float] = None, coef0: float = 0.0):
        """

            Input:
                C: regularization coefficient. The strength of the regularization is inversely proportional to C
                kernel: type of kernel which will be applied to the input vectors
                degree: degree of the polynomial kernel function (‘poly’)
                gamma: kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                coef0: independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            Output:
                SupportVectorClassifier instance
        """

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        # choose kernel for kernel trick

        if(self.kernel == 'linear'):
            self.kernel_function = linear()
        elif(self.kernel == 'poly'):
            self.kernel_function = polynomial(degree, gamma, coef0)
        elif(self.kernel == 'rbf'):
            self.kernel_function = rbf(gamma)
        elif(self.kernel == 'sigmoid'):
            self.kernel_function = sigmoid(degree, gamma, coef0)


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Find the optimal parameters

            We solve SVM with soft margin and use Karush — Kuhn — Tucker conditions for solving
            non-linear optimization problem (quadratic problem)

            The CVXOPT library is used for optimizing parameters

            Input:
                X: 2d array of input features with shape (n_samples, n_features)
                y: labels for the given features with shape (n_samples, )
                   NOTE: y = 1 or -1, if you have 0, 1 convert it to -1, 1
            Output:
                fitted instance of SupportVectorClassifier

        """

        n_samples, n_features = X.shape

        if (self.gamma is None):
            self.gamme =  1 / n_features


        # precalculate all the pairs of X with applied kernel function
        # kernel function can be considered as a dot product of <f(X1), f(X2)>
        # where f() convert X somehow for example f(X) = X^2
        # we use it instead of simple <X1, X2> dot product, it is also known as a Kernel Trick
        # and also right here we calculate P matrix for cvxopt

        P = np.zeros(shape = (n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                P[i, j] = y[i] * y[j] * self.kernel_function(X[i], X[j])

        q = np.full(shape=(n_samples, 1), fill_value=-1)
        G = np.eye(N = n_samples) * -1
        h = np.zeros(shape=(n_samples, 1))
        A = y.copy().reshape((1, n_samples))
        b = 0

        # if we build SVM with Soft Margin and have C - regularization coefficient
        # we need to add new constrains to the matrices G and h because we have 2 inequality now
        # in the part above we described: -a <= 0
        # in this part we add information about: a <= C
        # where a is a list of lagrange cofficients
        if (self.C is not None):
            G = np.vstack((G, np.eye(N = n_samples)))
            h = np.vstack((h, np.full(shape=(n_samples, 1), fill_value=self.C)))


        # convert matrices to the cvxopt format
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q, tc='d')
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A, tc='d')
        b = cvxopt.matrix(b, tc='d')

        # run cvxopt quadratic problem solver
        solution_parameters = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.lagrange_coefficients = np.ravel(solution_parameters['x'])

        # once we have lagrange coefficient we can get support vectors and find vector of weights and bias
        # NOTE: support vectors are points in X where 0 < lagrange_coeff < C
        #       and we also have "penalty" support vectors in X where lagrange_coeff = C, such points prevent
        #       the task from linear separability but they are still considered as support vectors
        self.support_ = np.nonzero(self.lagrange_coefficients > 0.0001)[0]
        self.lagrange_coefficients = self.lagrange_coefficients[self.support_]
        self.support_vectors_ = X[self.support_]
        self.support_vector_labels = y[self.support_]

        # we know a formula for SVC prediction ans(x) = sign( for i [0, l](alfa[i] * y[i] * kernel(x, x[i])) - b)
        # so lets get coefficient 'b' from the formula and use our support vectors and lagrange coeffs to calculate bias for all
        # support vector and after we take median for all biases
        support_vector_biases = []
        for i in range(len(self.support_)):
            current_bias = self.support_vector_labels[i]
            for j in range(len(self.support_)):
                current_bias -= self.lagrange_coefficients[j] * self.support_vector_labels[j] \
                                * self.kernel_function(self.support_vectors_[i], self.support_vectors_[j])
            support_vector_biases.append(current_bias)

        self.bias = np.median(support_vector_biases)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Make prediction to the given set of input features

            Input:
                X: 2d array of input features with shape (n_samples, n_features)
            Output:
                predicted labels for each input sample with shape (n_samples, )
                output values are -1 or 1
        """

        y_pred = []

        for x in X:
            support_vector_values = self.bias
            for i in range(len(self.support_)):
                support_vector_values += self.lagrange_coefficients[i] * self.support_vector_labels[i] \
                                       * self.kernel_function(self.support_vectors_[i], x)
            y_pred.append(np.sign(support_vector_values))

        return np.array(y_pred)

#TODO: add regularization + apply function to the input values kind of kernel transformation
class SupportVectorClassifierSGD:
    """
        Support Vector Classifier with Hinge Loss + SGD optimization

        Links:
            1. https://habr.com/ru/company/ods/blog/484148/ [RU] (Hinge Loss + Gradient Descent)
            2. https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC (very close but here we can use non-linear transformations)

    """

    def __init__(self, loss: Callable = HingeLoss(), max_iter: int = 1000, learning_rate: float = 0.01):
        """
            Input:
                loss: function which we minimize. Hinge Loss is default for SVM
                max_iter: number of iterations during optimization
                learning_rate: how strong we change parameters on each optimization step
            Output:
                SupportVectorClassifierSGD object
        """
        self.loss = loss
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Iteratively change SVM parameters for the best fit with loss function
            Usually we optimize Hinge Loss with SGD (Stochastic Gradient Descent) as on optimization algorithm

            Input:
                X: input variables with shape (number_samples, number_features)
                y: target values with shape (number_samples, ) and values equal 1 or -1
            Output:
                self
        """

        X_ = np.insert(X, 0, 1, axis=1)

        self.weights = np.random.rand(X_.shape[1])  # just a random initialization in [0, 1) values range

        for iter_ind in range(self.max_iter):

            y_pred = self.predict(X)

            loss_gradient = self.loss.gradient(y, y_pred)

            #TODO: it doesn't take into account that y = sign(x * w + b), we consider it like y = x * w + b
            #      is it correct???
            weights_gradient = loss_gradient @ X_  # we use the chain rule to get loss gradient with respect to each weight


            #TODO: add regulariztion for the weights
            #add gradient from regularization function if it's defined
            #if (self.regularization_function is not None):
            #    weights_gradient += self.regularization_function.gradient(self.weights)

            self.weights -= weights_gradient * self.learning_rate

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Predict values for the input features

            Input:
                X: input variable with shape (number_sample, number_features)
            Output:
                predictions for each sample with shape (number_samples, )
                each prediction can be only 1,0,-1
        """

        # add a fake first column with ones instead of using separate bias variable
        X_ = np.insert(X, 0, 1, axis=1)

        return np.sign(X_ @ self.weights)
