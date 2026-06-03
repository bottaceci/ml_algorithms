import numpy as np

def one_hot_encode(y, n_classes=None):
    y = np.asarray(y).astype(int).ravel()

    if n_classes is None:
        n_classes = np.max(y) + 1

    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1

    return one_hot