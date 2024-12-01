import family

import numpy as np


class SGD:
    def __init__(
        self,
        family: family.Family,
        lr: float,
        batch_size: int,
        method: str = "adam",
        epochs: int = 1000,
        tol: float = 1e-16,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
    ):
        self.family = family
        self.lr0 = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        step_dict = {
            "momentum": self.momentum,
            "adam": self.adam,
        }
        self.step = step_dict[method]

    def predict(self, X):
        return self.family.inverse_link(np.dot(X, self.params))

    def loss(self, y, y_pred):
        return -self.family.loglike(y, y_pred, self.dispersion)

    def gradient(self, X, y):
        y_pred = self.predict(X)
        return -self.family.gradient_loglike(X, y, y_pred, self.dispersion)

    def update_params(self, grad):
        step = self.step(grad)
        self.params += step[: self.n_features]
        self.dispersion += step[-1]

    def momentum(self, grad):
        self.v = self.beta1 * self.v - (1 - self.beta1) * self.lr * grad
        return self.v

    def adam(self, grad):
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.G = self.beta2 * self.G + (1 - self.beta2) * grad**2
        return -self.lr / np.sqrt(self.G + self.eps) * self.v

    def fit(self, X, y):
        # number of individuals and parameters
        self.n_obs, self.n_features = X.shape

        # initial values
        self.params = np.random.uniform(0, 1, self.n_features)
        self.dispersion = 1

        y_pred = self.predict(X)
        best_loss = self.loss(y, y_pred)
        self.loss_history = np.array([])

        # init momentum and adaptive gradient auxiliary arrays with zeros
        self.v = np.zeros(self.n_features + 1)
        self.G = np.zeros(self.n_features + 1)

        self.lr = self.lr0
        epochs_no_change, self.n_iter = 0, 0
        params_prev = np.append(self.params, self.dispersion)
        while self.n_iter < self.epochs:
            # shuffle data
            idx = np.random.permutation(self.n_obs)
            X, y = X[idx], y[idx]
            
            for start in range(0, self.n_obs, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                grad = self.gradient(X_batch, y_batch)
                self.update_params(grad)

            # update loss after each epoch
            y_pred = self.predict(X)
            loss = self.loss(y, y_pred)
            self.loss_history = np.append(self.loss_history, loss)

            # each time 2 consecutive epochs fail to decrease loss by tol current learning rate is divided by 5
            if loss > best_loss - self.tol:
                epochs_no_change += 1
                if epochs_no_change == 2:
                    self.lr /= 5
                    epochs_no_change = 0
            else:
                best_loss = loss
                epochs_no_change = 0

            # stop criterion
            params = np.append(self.params, self.dispersion)
            if np.linalg.norm(params - params_prev) < self.tol:
                break
            params_prev = params

            self.n_iter += 1

        return self

