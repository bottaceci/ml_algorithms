from time import perf_counter
import numpy as np
import metrics.regression as rg
from itertools import product

def parameter_sweep(model_cls, param_grid, X_train, y_train, X_test, y_test, 
                    base_params=None):
    base_params = base_params or {}
    sweep_results = []
    loss_curves = []

    keys = list(param_grid.keys())

    for values in product(*param_grid.values()):
        sweep_params = dict(zip(keys,values))
        params = {**base_params, **sweep_params}

        model = model_cls(**params) 

        start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - start

        loss_array = np.array(model.loss_history)

        is_stable = np.all(np.isfinite(loss_array))

        if is_stable:
            preds = model.predict(X_test)

            test_mse = rg.mean_squared_error(y_test, preds)
            test_rmse = rg.root_mean_squared_error(y_test, preds)
            test_mae = rg.mean_absolute_error(y_test, preds)
            test_r2 = rg.r2_score(y_test, preds)
        else:
            test_mse = np.nan
            test_rmse = np.nan
            test_mae = np.nan
            test_r2 = np.nan

        converged = (
            np.all(np.isfinite(loss_array))
            and loss_array[-1] < loss_array[0]
        )

        sweep_results.append({
            **sweep_params,
            **base_params,
            "final_train_loss": model.loss_history[-1],
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "fit_time": fit_time,
            "initial_train_loss": loss_array[0],
            "final_train_loss": loss_array[-1],
            "min_train_loss": loss_array.min(),
            "stable": is_stable,
            "converged": converged,
        })

        loss_curves.append({
            **sweep_params,
            'loss': loss_array
        })

    return sweep_results, loss_curves