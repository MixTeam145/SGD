import numpy as np
from scipy.special import gamma, digamma


class Family:
    def __init__(self):
        self.name = None

    def link(self, x):
        raise NotImplementedError()

    def inverse_link(self, x):
        raise NotImplementedError()

    def loglike(self, y, y_pred, dispersion):
        raise NotImplementedError()

    def gradient_loglike(self, X, y, y_pred, dispersion):
        raise NotImplementedError()

    def deviance(self, y, y_pred):
        raise NotImplementedError()


class Gaussian(Family):
    def __init__(self):
        self.name = "Gaussian"

    def link(self, x):
        return x

    def inverse_link(self, x):
        return x

    def loglike(self, y, y_pred, dispersion):
        n = len(y)
        error = y - y_pred
        rss = np.sum(error**2)
        return -n / 2 * np.log(2 * np.pi * dispersion) - rss / (2 * dispersion)

    def gradient_loglike(self, X, y, y_pred, dispersion):
        n = len(y)
        error = y - y_pred
        rss = np.sum(error**2)
        return np.append(
            np.dot(X.T, error) / dispersion,
            -n / (2 * dispersion) + rss / (2 * dispersion**2),
        )

    def deviance(self, y, y_pred):
        error = y - y_pred
        return np.sum(error**2)


class Gamma(Family):
    def __init__(self):
        self.name = "Gamma"

    def link(self, x):
        return np.log(x)

    def inverse_link(self, x):
        return np.exp(x)

    def loglike(self, y, y_pred, dispersion):
        n = len(y)
        k = 1 / dispersion
        return -n * np.log(gamma(k)) + np.sum(
            k * np.log(k * y / y_pred) - np.log(y) - k * y / y_pred
        )

    def gradient_loglike(self, X, y, y_pred, dispersion):
        n = len(y)
        k = 1 / dispersion
        error = y - y_pred
        return np.append(
            k * np.dot(X.T, error / y_pred),
            k**2 * (n * digamma(k) - n + np.sum(-np.log(k * y / y_pred) + y / y_pred)),
        )

    def deviance(self, y, y_pred):
        return 2 * np.sum(-np.log(y / y_pred) + (y - y_pred) / y_pred)
