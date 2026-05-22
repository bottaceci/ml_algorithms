from time import perf_counter
import numpy as np
import metrics.regression as rg
import metrics.classification as cl
from itertools import product

def parameter_sweep_regression(model_cls, param_grid, X_train, y_train, X_test, y_test, 
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

def parameter_sweep_classification(model_cls, param_grid, X_train, y_train, X_test, y_test, 
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
            proba = model.predict_proba(X_test)

            accuracy = cl.accuracy_score(y_test, preds)
            precision = cl.precision_score(y_test, preds)
            recall = cl.recall_score(y_test, preds)
            f1 = cl.f1_score(y_test, preds)
            rocauc = cl.roc_auc_score(y_test, proba)
        else:
            accuracy = np.nan
            precision = np.nan
            recall = np.nan
            f1 = np.nan
            rocauc = np.nan

        converged = (
            np.all(np.isfinite(loss_array))
            and loss_array[-1] < loss_array[0]
        )

        sweep_results.append({
            **sweep_params,
            **base_params,
            "final_train_loss": model.loss_history[-1],
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": rocauc,
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

def threshold_sweep_classification(
    model,
    X_test,
    y_test,
    thresholds,
):
    proba = model.predict_proba(X_test)

    results = []

    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "test_accuracy": cl.accuracy_score(y_test, preds),
            "test_precision": cl.precision_score(y_test, preds),
            "test_recall": cl.recall_score(y_test, preds),
            "test_f1": cl.f1_score(y_test, preds),
            "test_roc_auc": cl.roc_auc_score(y_test, proba),
        })

    return results

def learning_rate_threshold_sweep(
    model_cls,
    lr_values,
    threshold_values,
    X_train,
    y_train,
    X_test,
    y_test,
    base_params=None,
):
    base_params = base_params or {}

    results = []
    loss_curves = []

    for lr in lr_values:
        params = {**base_params, "lr": lr}
        model = model_cls(**params)

        start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - start

        loss_array = np.array(model.loss_history)
        is_stable = np.all(np.isfinite(loss_array))

        if is_stable:
            proba = model.predict_proba(X_test)

            for threshold in threshold_values:
                preds = (proba >= threshold).astype(int)

                results.append({
                    "lr": lr,
                    "threshold": threshold,
                    **base_params,
                    "test_accuracy": cl.accuracy_score(y_test, preds),
                    "test_precision": cl.precision_score(y_test, preds),
                    "test_recall": cl.recall_score(y_test, preds),
                    "test_f1": cl.f1_score(y_test, preds),
                    "test_roc_auc": cl.roc_auc_score(y_test, proba),
                    "fit_time": fit_time,
                    "initial_train_loss": loss_array[0],
                    "final_train_loss": loss_array[-1],
                    "min_train_loss": loss_array.min(),
                    "stable": True,
                    "converged": loss_array[-1] < loss_array[0],
                })
        else:
            for threshold in threshold_values:
                results.append({
                    "lr": lr,
                    "threshold": threshold,
                    **base_params,
                    "test_accuracy": np.nan,
                    "test_precision": np.nan,
                    "test_recall": np.nan,
                    "test_f1": np.nan,
                    "test_roc_auc": np.nan,
                    "fit_time": fit_time,
                    "initial_train_loss": loss_array[0],
                    "final_train_loss": loss_array[-1],
                    "min_train_loss": np.nan,
                    "stable": False,
                    "converged": False,
                })

        loss_curves.append({
            "lr": lr,
            "loss": loss_array,
        })

    return results, loss_curves

def parameter_sweep_trees(model_cls, param_grid, X_train, y_train, X_test, y_test, 
                    base_params=None):
    base_params = base_params or {}
    sweep_results = []

    keys = list(param_grid.keys())

    for values in product(*param_grid.values()):
        sweep_params = dict(zip(keys,values))
        params = {**base_params, **sweep_params}

        model = model_cls(**params) 

        start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - start

        preds = model.predict(X_test)

        accuracy = cl.accuracy_score(y_test, preds)
        precision = cl.precision_score(y_test, preds)
        recall = cl.recall_score(y_test, preds)
        f1 = cl.f1_score(y_test, preds)

        sweep_results.append({
            **sweep_params,
            **base_params,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "fit_time": fit_time,
        })

    return sweep_results