import numpy as np
from typing import Optional
from mlzero.metrics.kernel_functions import *
import cvxopt

# Turn off CVXOPT output
cvxopt.solvers.options['show_progress'] = False

class SupportVectorClassifier:
    """
    Support Vector Classifier base on QP(Quadratic Problem) optimization with CVXOPT library

    Links:

        1. https://blog.dominodatalab.com/fitting-support-vector-machines-quadratic-programming [ENG] (QP optimization)
        2. https://neerc.ifmo.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D1%85_%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2_(SVM) [RU] (QP optimization)
        3. https://habr.com/ru/company/ods/blog/484148/ [RU] (Hinge Loss + Gradient Descent)
        4. https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8 [ENG] (QP optimization)
        5. https://xavierbourretsicotte.github.io/SVM_implementation.html [ENG] (QP using CVXOPT)
    """


    def __init__(self, C: float = 1.0, kernel: str = 'linear', degree: int = 3, gamma: Optional[float] = None,
                 coef0: float = 0.0):

        """

            Input:
                C:
                kernel:
                degree:
                gamma:
                coef0:
            Output:
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

            Input:
                X:
            Output:
        """

        y_pred = []

        for x in X:
            support_vector_values = self.bias
            for i in range(len(self.support_)):
                support_vector_values += self.lagrange_coefficients[i] * self.support_vector_labels[i] \
                                       * self.kernel_function(self.support_vectors_[i], x)
            y_pred.append(np.sign(support_vector_values))

        return np.array(y_pred)