import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Abstract Layer class
class Layer(ABC):
    @abstractmethod
    def forward(self, X: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, grad: Array) -> Array:
        pass

    def update(self, lr: float) -> None:
        pass

    def reset_parameters(self) -> None:
        pass

# Linear layer
class Linear(Layer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        init_method: str = 'xavier', 
        random_state:int | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.random_state = random_state

        self.W: Array
        self.b: Array

        self.X: Array | None = None
        self.dW: Array | None = None
        self.db: Array | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng = np.random.default_rng(self.random_state)

        if self.init_method == 'xavier':
            limit = np.sqrt(6 / (self.in_features + self.out_features))
            self.W = rng.uniform(-limit, limit, size=(self.in_features, self.out_features))
        elif self.init_method == 'normal':
            self.W = rng.normal(0, 0.01, size=(self.in_features, self.out_features))
        else:
            raise ValueError('Please use a known initialization mode (normal, xavier).')
        
        self.b: Array = np.zeros((1, self.out_features))

        self.X: Array | None = None
        self.dW: Array | None = None
        self.db: Array | None = None

    def forward(self, X: Array) -> Array:
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, grad: Array) ->  Array:
        if self.X is None:
            raise ValueError("Cannot call backward before forward.")
        
        self.dW = self.X.T @ grad / self.X.shape[0]
        self.db = np.sum(grad, axis=0, keepdims=True) / self.X.shape[0]

        return grad @ self.W.T
    
    def update(self, lr: float) -> None:
        if self.dW is None or self.db is None:
            return
        
        self.W -= lr * self.dW
        self.b -= lr * self.db

# Activation functions
class Tanh(Layer):
    def __init__(self):
        self.A: Array | None = None

    def forward(self, X: Array) -> Array:
        self.A = np.tanh(X)
        return self.A
    
    def backward(self, grad: Array) -> Array:
        if self.A is None:
            raise ValueError("Cannot call backward before forward.")
        
        return grad * (1 - self.A ** 2)
    
    def update(self, lr):
        pass

    def reset_parameters(self):
        return super().reset_parameters()

class Sigmoid(Layer):
    def __init__(self):
        self.A: Array | None = None

    def forward(self, X: Array) -> Array:
        X = np.clip(X, -500, 500)
        self.A = 1 / (1 + np.exp(-X))
        return self.A
    
    def backward(self, grad: Array) -> Array:
        if self.A is None:
            raise ValueError("Cannot call backward before forward.")
        
        return grad * self.A * (1 - self.A)
    
    def update(self, lr):
        pass

    def reset_parameters(self):
        return super().reset_parameters()
    
class Softmax(Layer):
    def __init__(self):
        self.output: Array | None = None

    def forward(self, X: Array) -> Array:
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad: Array) -> Array:
        # Usually, when Softmax is paired with categorical cross-entropy,
        # the combined gradient is handled directly by the loss:
        # dZ = y_pred - y_true.
        return grad
    
    def reset_parameters(self):
        return super().reset_parameters()

# Loss
class BinaryCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1, 1)

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1-eps)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )
    
    def backward(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1, 1)

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -(
            y_true / y_pred
            - (1 - y_true) / (1 - y_pred)
        )
    
class CategoricalCrossEntropy:
    def forward(self, y_true: Array, y_pred: Array) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true: Array, y_pred: Array) -> Array:
        return y_pred - y_true

# Model container
class Sequential:
    def __init__(self, *layers: Layer, task: str = 'binary') -> None:
        self.layers: tuple[Layer, ...] = layers
        self.task = task
        self.loss_history: list[float] = []
        self.current_epoch: int = 0

        if self.task not in {"binary", "multiclass"}:
            raise ValueError("task must be either 'binary' or 'multiclass'")

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

        self.loss_history = []
        self.current_epoch = 0

    def forward(self, X: Array) -> Array:
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output
    
    def backward(self, grad: Array) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr: float) -> None:
        for layer in self.layers:
            layer.update(lr)

    def fit(
        self, 
        X: Array, 
        y: Array, 
        loss_fn, 
        lr: float = 0.1, 
        n_iters: int = 1000, 
        reset: bool = True
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.task == 'binary':
            y = y.reshape(-1, 1)

        if reset:
            self.reset_parameters()

        for _ in range(n_iters):
            y_pred = self.forward(X)

            loss = loss_fn.forward(y, y_pred)
            self.loss_history.append(loss)

            grad = loss_fn.backward(y, y_pred)
            self.backward(grad)
            self.update(lr)

            self.current_epoch += 1

        return self
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        proba = self.predict_proba(X)

        if self.task == "binary":
            return (proba.ravel() >= threshold).astype(int)

        if self.task == "multiclass":
            return np.argmax(proba, axis=1)

        raise ValueError(f"Unknown task: {self.task}")