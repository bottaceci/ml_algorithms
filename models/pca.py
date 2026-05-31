import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X): # don't need class labels because this is unsupervised training
        X = np.asarray(X)
        
        # mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # covariance, cov function needs samples as columns -> transpose
        cov = np.cov(X_centered.T)

        # eigenvectors, eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov) # used np.linalg.eigh instead of eig because covariance matrix is symmetric

        # sort eigenvectors
        idxs = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idxs]
        eigvecs = eigvecs[:, idxs]

        # store the first n_components eigenvectors 
        self.components = eigvecs[:, :self.n_components].T
        self.explained_variance = eigvals[:self.n_components]
        self.explained_variance_ratio = (
            eigvals[:self.n_components] / np.sum(eigvals)
        )


    def transform(self, X):
        if self.components is None:
            raise ValueError("PCA has not been fitted yet.")

        X = np.asarray(X)
        X_centered = X - self.mean

        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)