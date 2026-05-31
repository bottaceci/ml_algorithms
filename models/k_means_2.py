import numpy as np


def squared_euclidean_distance(x1, x2):
    return np.sum((x1 - x2) ** 2)


class KMeans:
    def __init__(self, K=5, max_iters=100, tol=1e-4, random_state=None):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia_history = []

    def fit(self, X):
        X = np.asarray(X)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        random_sample_idxs = rng.choice(n_samples, self.K, replace=False)
        self.centroids = X[random_sample_idxs].copy()
        self.inertia_history = []

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)

            inertia = self._compute_inertia(X, labels)
            self.inertia_history.append(inertia)

            shift = np.sum(
                np.sqrt(np.sum((self.centroids - new_centroids) ** 2, axis=1))
            )

            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels = self._assign_clusters(X)

        return self

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("KMeans has not been fitted yet.")

        X = np.asarray(X)
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.K))

        for k, centroid in enumerate(self.centroids):
            distances[:, k] = np.sum((X - centroid) ** 2, axis=1)

        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.K, n_features))

        for k in range(self.K):
            cluster_points = X[labels == k]

            if len(cluster_points) == 0:
                centroids[k] = self.centroids[k]
            else:
                centroids[k] = np.mean(cluster_points, axis=0)

        return centroids

    def _compute_inertia(self, X, labels):
        inertia = 0.0

        for k in range(self.K):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)

        return inertia