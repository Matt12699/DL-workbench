from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)
