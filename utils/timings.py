from time import perf_counter
import numpy as np
import torch
import torch.nn as nn

def benchmark_model(model_factory, X_train, y_train, X_test, n_runs=30):
    fit_times = []
    predict_times = []

    for _ in range(n_runs):
        model = model_factory()

        start = perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(perf_counter() - start)

        start = perf_counter()
        preds = model.predict(X_test)
        predict_times.append(perf_counter() - start)

    return {
        "fit_time_mean": np.mean(fit_times),
        "fit_time_std": np.std(fit_times),
        "predict_time_mean": np.mean(predict_times),
        "predict_time_std": np.std(predict_times),
    }

def benchmark_pytorch_model(
        model_factory, 
        X_train, 
        y_train, 
        X_test, 
        n_iters,
        lr,
        n_runs=30, 
        criterion_cls = nn.MSELoss,
        optimizer_cls = torch.optim.SGD,
        predict_fn = None
    ):
    fit_times = []
    predict_times = []

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    for _ in range(n_runs):
        torch.manual_seed(42)

        torch_model = model_factory()
        
        with torch.no_grad():
            torch_model.weight.zero_()
            torch_model.bias.zero_()

        criterion = criterion_cls()
        optimizer = optimizer_cls(torch_model.parameters(), lr=lr)

        start = perf_counter()

        for _ in range(n_iters):
            y_pred = torch_model(X_train_tensor)
            loss = criterion(y_pred, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fit_times.append(perf_counter() - start)

        start = perf_counter()

        with torch.no_grad():
            if predict_fn:
                torch_preds = predict_fn(X_test_tensor)
            else:
                torch_preds = torch_model(X_test_tensor).numpy().ravel()

        predict_times.append(perf_counter() - start)

    return {
        "fit_time_mean": np.mean(fit_times),
        "fit_time_std": np.std(fit_times),
        "predict_time_mean": np.mean(predict_times),
        "predict_time_std": np.std(predict_times),
    }