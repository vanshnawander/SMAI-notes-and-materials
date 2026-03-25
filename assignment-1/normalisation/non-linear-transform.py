import math 
import numpy as np


class LogTransformation:
    def __init__(self):
        self.c_ = 0 # constant to avoid log(0)

    def fit(self, X, c=0):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.c_ = c

    def transform(self, X):
        if np.any(X <= self.c_):
            raise ValueError("Log transformation is only defined for positive values.")
        return np.log(X + self.c_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class ReciprocalTransformation:
    """
    To avaoid division by zero error as mentioned c can be set to small value like 1e-9 or something.
    """
    def __init__(self):
        self.c_ = 0 # constant to avoid division by zero 

    def fit(self, X, c=0):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.c_ = c

    def transform(self, X):
        if np.any(X + self.c_ == 0):
            raise ValueError("Reciprocal transformation is only defined for non-zero values.")
        return 1 / (X + self.c_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class SquareRootTransformation:
    def __init__(self):
        pass 

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        if np.any(X < 0):
            raise ValueError("Square root transformation is only defined for non-negative values.")
        return np.sqrt(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class BoxCoxTransformation:
    def __init__(self):
        self.lmbda_ = 0.5 # default value for lambda

    def fit(self, X, lmbda=0.5):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        if np.any(X <= 0):
            raise ValueError("Box-Cox transformation is only defined for strictly positive values.")
        self.lmbda_ = lmbda

    def transform(self, X):
        if self.lmbda_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        if self.lmbda_ == 0:
            return np.log(X)
        else:
            return (X ** self.lmbda_ - 1) / self.lmbda_

    def fit_transform(self, X, lmbda=0.5):
        self.fit(X, lmbda)
        return self.transform(X)
    

class YeoJohnsonTransformation:
    def __init__(self):
        self.lmbda_ = 0.5 # default value for lambda

    def fit(self, X, lmbda=0.5):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.lmbda_ = lmbda

    def transform(self, X):
        if self.lmbda_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        transformed = np.zeros_like(X)
        for i in range(len(X)):
            if X[i] >= 0:
                if self.lmbda_ == 0:
                    transformed[i] = np.log(X[i] + 1)
                else:
                    transformed[i] = ((X[i] + 1) ** self.lmbda_ - 1) / self.lmbda_
            else:
                if self.lmbda_ == 2:
                    transformed[i] = -np.log(-X[i] + 1)
                else:
                    transformed[i] = -((-X[i] + 1) ** (2 - self.lmbda_) - 1) / (2 - self.lmbda_)
        return transformed

    def fit_transform(self, X, lmbda=0.5):
        self.fit(X, lmbda)
        return self.transform(X)
    

class HyperbolicTangentTransformation:
    def __init__(self):
        pass

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        return np.tanh(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class LogisticTransformation:
    def __init__(self):
        pass

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."

    def transform(self, X):
        return 1 / (1 + np.exp(-X))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)