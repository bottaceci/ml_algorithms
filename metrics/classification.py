import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, pos_label=1, average='binary'):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

        return 0.0 if tp + fp == 0 else tp / (tp + fp)

    if average == 'macro':
        classes = np.unique(y_true)
        scores = [
            precision_score(y_true, y_pred, pos_label=c, average='binary')
            for c in classes
        ]
        return np.mean(scores)
    
    raise ValueError("average must be 'binary' or 'macro'")


def recall_score(y_true, y_pred, pos_label=1, average='binary'):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        return 0.0 if tp + fn == 0 else tp / (tp + fn)

    if average == "macro":
        classes = np.unique(y_true)
        scores = [
            recall_score(y_true, y_pred, pos_label=c, average="binary")
            for c in classes
        ]
        return np.mean(scores)

    raise ValueError("average must be 'binary' or 'macro'")


def f1_score(y_true, y_pred, pos_label=1, average='binary'):
    if average == "binary":
        precision = precision_score(y_true, y_pred, pos_label=pos_label, average="binary")
        recall = recall_score(y_true, y_pred, pos_label=pos_label, average="binary")

        return 0.0 if precision + recall == 0 else (
            2 * precision * recall / (precision + recall)
        )

    if average == "macro":
        y_true = np.asarray(y_true).ravel()
        classes = np.unique(y_true)

        scores = [
            f1_score(y_true, y_pred, pos_label=c, average="binary")
            for c in classes
        ]

        return np.mean(scores)

    raise ValueError("average must be 'binary' or 'macro'")

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    positives = np.sum(y_true == 1)
    negatives = np.sum((y_true == 0) | (y_true == -1))

    if positives == 0 or negatives == 0:
        return np.nan

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum((y_true_sorted == 0) | (y_true_sorted == -1))

    tpr = tps / positives
    fpr = fps / negatives

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    return np.trapezoid(tpr, fpr)