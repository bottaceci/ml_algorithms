import numpy as np

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000): 
        #eventually also set the activation function as an initial 
        # parameter, putting it in an utils module
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features) 
        # it's better to use a random initialization insteas of zeros
        self.bias = 0

        # make sure the labels are all either 0 or 1
        y_ = np.where(y>0, 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x 
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred