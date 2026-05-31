import numpy as np
import utils.functions as F

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000, activation_func=F.unit_step_func): 
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = activation_func
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features) 
        # it's better to use a random initialization insteas of zeros
        self.bias = 0.0
        self.loss_history = []

        # make sure the labels are all either 0 or 1
        y_ = np.where(y>0, 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            n_errors = 0

            for idx, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_pred)

                if update != 0:
                    n_errors += 1

                self.weights += update * x 
                self.bias += update
            
            self.loss_history.append(n_errors)

            if n_errors == 0:
                break

        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)