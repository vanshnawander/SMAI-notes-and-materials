import math 
import numpy as np 


class ZscoreNormalization:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The normalizer has not been fitted yet.")
        if self.std_ == 0:
            return np.zeros_like(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class ModifiedZscoreNormalization:
    def __init__(self):
        self.median_ = None
        self.mad_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.median_ = np.median(X)
        self.mad_ = np.median(np.abs(X - self.median_))

    def transform(self, X):
        if self.median_ is None or self.mad_ is None:
            raise ValueError("The normalizer has not been fitted yet.")
        if self.mad_ == 0:
            return np.zeros_like(X)
        return 0.6745 * (X - self.median_) / self.mad_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class ParetoScaling:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The normalizer has not been fitted yet.")
        if self.std_ == 0:
            return np.zeros_like(X)
        return (X - self.mean_) / np.sqrt(self.std_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
