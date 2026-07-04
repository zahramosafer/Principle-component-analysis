import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components   # FIXED

        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # mean
        self.mean_ = np.mean(X, axis=0)

        # center data
        X_centered = X - self.mean_

        # covariance matrix
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

        # eigen decomposition
        eigval, eigvec = np.linalg.eigh(cov)

        # sort descending
        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        # store components
        self.components_ = eigvec[:, :self.n_components]

        # FIXED typo
        self.eigenvalues_ = eigval

        # explained variance ratio
        self.explained_variance_ratio_ = eigval / np.sum(eigval)

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)