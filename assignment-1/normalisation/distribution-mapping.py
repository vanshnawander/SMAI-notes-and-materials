import math
import numpy as np

class QuantileNormalization:
    def __init__(self):
        self.quantiles_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.quantiles_ = np.percentile(X, np.arange(0, 101))

    def transform(self, X):
        if self.quantiles_ is None:
            raise ValueError("The normalizer has not been fitted yet.")
        return np.interp(X, self.quantiles_, np.linspace(0, 1, len(self.quantiles_)))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class RankGaussNormalization: