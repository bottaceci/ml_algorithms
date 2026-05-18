import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X): # don't need class labels because this is unsupervised training
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, cov function needs samples as columns -> transpose
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigvecs, eigvals = np.linalg.eig(cov)

        # eigenvectors v = [:,i] originally column vector, transpose this for easier calculations
        eigvecs = eigvecs.T

        # sort eigenvectors
        idxs = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idxs]
        eigvecs = eigvecs[idxs]

        # store the first n_components eigenvectors 
        self.components = eigvecs[:self.n_components]


    def transform(self, X): # this method can receive either the training data or the new testing data
        # project the data
        X = X - self.mean
        return np.dot(X, self.components.T)