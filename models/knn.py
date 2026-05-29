import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def euclidean_distance_squared(x1, x2):
    distance = np.sum((x1-x2)**2)
    return distance

class KNN:
    def __init__(self, k, dist='euclidean'):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.dist = dist

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        return self

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # compute the distance
        if self.dist == 'euclidean':
            distances = np.array([
                euclidean_distance(x, x_train)
                for x_train in self.X_train
            ])
        elif self.dist == 'squared':
            distances = np.array([
                euclidean_distance_squared(x, x_train)
                for x_train in self.X_train
            ])
        else:
            "Use dist = 'euclidean' or 'squared'"

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # determine the label with majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]