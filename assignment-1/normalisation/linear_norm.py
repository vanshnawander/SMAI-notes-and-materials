import math 
import numpy as np

class LinearScaling:
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
