import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # verify labels
        y_ = np.where(y <= 0, -1, 1)

        # init the weights
        self.w = np.zeros(n_features) # best to random initialize
        self.b = 0

        # update w and b
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y_[idx]*(np.dot(x, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - y_[idx]*x)
                    self.b -= self.lr*y_[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)