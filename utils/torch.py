import torch
import numpy as np

class TorchModelWrapper:
    def __init__(
            self, 
            model, 
            loss_fn, 
            optimizer_cls, 
            lr,
            n_iters, 
            predict_fn,
            predict_proba_fn, 
            device="cpu", 
            zero_weights = False,
            seed = 42
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.n_iters = n_iters
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.device = device
        self.loss_history = []
        self.zero_weights = zero_weights
        self.seed = seed


    def fit(self, X, y):
        torch.manual_seed(self.seed)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        if y_tensor.ndim == 1:
            y_tensor = y_tensor.reshape(-1, 1)

        if self.zero_weights:
            with torch.no_grad():
                self.model.weight.zero_()
                self.model.bias.zero_()

            self.loss_history = []

        for _ in range(self.n_iters):
            y_pred = self.model(X_tensor)
            loss = self.loss_fn(y_pred, y_tensor)

            self.loss_history.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        if self.predict_fn is None:
            raise ValueError("No predict_fn was provided.")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(X_tensor)

        return self.predict_fn(output)

    def predict_proba(self, X):
        if self.predict_proba_fn is None:
            raise ValueError("No predict_proba_fn was provided.")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(X_tensor)

        return self.predict_proba_fn(output)