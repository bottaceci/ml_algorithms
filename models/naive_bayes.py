import numpy as np

class NaiveBayes:

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing # If a feature has near-zero variance inside a class, division by zero can break the Gaussian PDF.
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate the mean, the var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + self.var_smoothing
            self._priors[idx] = X_c.shape[0] / float(n_samples)

        return self

    def predict(self, X):
        if self._classes is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X)
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        posteriors = []

        # calculate the posterior probability for each class
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + likelihood)

        # return the class with the highest posterior
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
