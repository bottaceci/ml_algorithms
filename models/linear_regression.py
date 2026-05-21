import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            loss = np.mean((y_pred - y)**2)
            self.loss_history.append(loss)

            dw = (2/n_samples)*np.dot(X.T, y_pred - y)
            db = (2/n_samples)*np.sum(y_pred - y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        X = np.asarray(X)

        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet")
        
        return np.dot(X, self.weights) + self.bias