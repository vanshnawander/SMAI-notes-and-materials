import math 
import numpy as np

class L1Norm:
    def __init__(self):
        pass

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        norm = np.sum(np.abs(X))
        if norm == 0:
            return np.zeros_like(X)
        return X / norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class L2Norm:
    def __init__(self):
        pass

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        norm = np.sqrt(np.sum(X ** 2))
        if norm == 0:
            return np.zeros_like(X)
        return X / norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class SoftmaxScaling:
    def __init__(self):
        pass

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        exp_X = np.exp(X)  
        return exp_X / np.sum(exp_X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)