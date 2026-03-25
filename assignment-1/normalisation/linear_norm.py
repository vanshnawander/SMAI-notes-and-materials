import math 
import numpy as np

class MinMaxScaling:
    """bounds output to 0,1"""
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.data_min_ = np.min(X)
        self.data_max_ = np.max(X)

    def transform(self, X):
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_)
        return self.feature_range[0] + (X - self.data_min_) * scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class MeanNormalization:
    def __init__(self):
        self.mean_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.mean_ = np.mean(X)
        self.data_min_ = np.min(X)
        self.data_max_ = np.max(X)

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("The normalizer has not been fitted yet.")
        if self.data_max_ - self.data_min_ == 0:
            return np.zeros_like(X)
        return (X - self.mean_) / (self.data_max_ - self.data_min_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MaxAbsoluteScaling:
    def __init__(self):
        self.max_abs_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.max_abs_ = np.max(np.abs(X))

    def transform(self, X):
        if self.max_abs_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        if self.max_abs_ == 0:
            return np.zeros_like(X)
        return X / self.max_abs_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class DecimalScaling:
    def __init__(self):
        self.j_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        max_abs = np.max(np.abs(X))
        self.j_ = math.ceil(math.log10(max_abs + 1)) if max_abs != 0 else 0

    def transform(self, X):
        if self.j_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        return X / (10 ** self.j_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class RobustScaling:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        assert len(X.shape) == 1, "Input data must be a 1D array."
        self.median_ = np.median(X)
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        self.iqr_ = q3 - q1

    def transform(self, X):
        if self.median_ is None or self.iqr_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        if self.iqr_ == 0:
            return np.zeros_like(X)
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
