import numpy as np
import utils.functions as F

def binary_cross_entropy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss

class MLP:
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            output_size=1, 
            random_state=None, 
            hidden_activation='sigmoid',
            lr=0.1, 
            n_iters=1000,
            threshold=0.5,
            initialization='random'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.random_state = random_state

        if hidden_activation == "sigmoid":
            self.activation = F.sigmoid
            self.activation_derivative = F.sigmoid_derivative

        elif hidden_activation == "tanh":
            self.activation = F.tanh
            self.activation_derivative = F.tanh_derivative

        else:
            raise ValueError('Please use a known activation function.')
        
        self.lr = lr
        self.n_iters = n_iters
        self.threshold = threshold
        self.initialization = initialization

        self.loss_history = []

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _initialize_parameters(self, init):
        rng = np.random.default_rng(self.random_state)

        if init == 'random':
            self.W1 = rng.normal(0, 0.01, size=(self.input_size, self.hidden_size))
            self.b1 = np.zeros((1, self.hidden_size))

            self.W2 = rng.normal(0, 0.01, size=(self.hidden_size, self.output_size))
            self.b2 = np.zeros((1, self.output_size))
        elif init == 'xavier':
            limit1 = np.sqrt(6 / (self.input_size + self.hidden_size))
            self.W1 = rng.uniform(-limit1, limit1, size=(self.input_size, self.hidden_size))
            self.b1 = np.zeros((1, self.hidden_size))

            limit2 = np.sqrt(6 / (self.hidden_size + self.output_size))
            self.W2 = rng.uniform(-limit2, limit2, size=(self.hidden_size, self.output_size))
            self.b2 = np.zeros((1, self.output_size))
        else:
            raise ValueError('Please use a known initialization mode.')

    def forward(self, X):
        X = np.asarray(X)

        Z1 = X @ self.W1 + self.b1
        A1 = self.activation(Z1)

        Z2 = A1 @ self.W2 + self.b2
        A2 = F.sigmoid(Z2)

        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
        }

        return A2, cache
    
    def backward(self, X, y, cache):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        m = X.shape[0]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - y
        dW2 = (1 / m) * (A1.T @ dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(A1)

        dW1 = (1 / m) * (X.T @ dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

        return grads
    
    def update_parameters(self, grads, lr):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]

        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def fit(self, X, y, snapshot_iters=None):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        self._initialize_parameters(self.initialization)
        self.loss_history = []

        snapshots = {}

        if snapshot_iters is None:
            snapshot_iters = []

        for i in range(self.n_iters):
            A2, cache = self.forward(X)

            loss = binary_cross_entropy(y, A2)
            self.loss_history.append(loss)

            grads = self.backward(X, y, cache)
            self.update_parameters(grads, self.lr)

            if i in snapshot_iters:
                snapshots[i] = {
                    "W1": self.W1.copy(),
                    "b1": self.b1.copy(),
                    "W2": self.W2.copy(),
                    "b2": self.b2.copy(),
                }

        return snapshots
    
    def predict_proba(self, X):
        A2, _ = self.forward(X)
        return A2.ravel()


    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)