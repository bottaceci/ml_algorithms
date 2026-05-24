import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 1000, threshold = 0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.threshold = threshold
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) + self.bias
            y_proba = sigmoid(linear_output)

            eps = 1e-15
            loss = -np.mean(y * np.log(y_proba + eps) + (1 - y) * np.log(1 - y_proba + eps))
            self.loss_history.append(loss)

            dw = (1/n_samples)*np.dot(X.T, y_proba - y)
            db = (1/n_samples)*np.sum(y_proba - y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

        return self

    def predict_proba(self,X):
        X = np.asarray(X)

        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        
        linear_output = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_output)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return (y_proba >= self.threshold).astype(int)
