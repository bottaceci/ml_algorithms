import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape

        # init the weights
        self.w = np.zeros(n_features) # best to random initialize
        self.b = 0.0
        self.loss_history = []

        # update w and b
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                margin = y_[idx]*(np.dot(x, self.w) - self.b)

                if margin >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2*self.lambda_param*self.w - y_[idx]*x
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

            margins = y_ * (np.dot(X, self.w) - self.b)
            hinge_losses = np.maximum(0, 1 - margins)
            loss = self.lambda_param * np.sum(self.w ** 2) + np.mean(hinge_losses)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)

        if self.w is None or self.b is None:
            raise ValueError("Model has not been fitted yet.")

        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0, 1, -1)