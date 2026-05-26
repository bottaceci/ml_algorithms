import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred, pos_label=1):
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

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