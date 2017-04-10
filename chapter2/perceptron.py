import numpy as np


class Perceptron(object):
    """
        Perceptron classifier
    """

    def __init__(self, eta=0.01, n_iter=10):
        """

        :param eta:
        :param n_iter:
        :return:
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
   