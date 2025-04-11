import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_means_ = None

    def fit(self, X, y=None):
        # Creo una copia per non modificare l'input originale
        X_copy = X.copy()
        
        # Sostituisco gli infiniti con NaN
        X_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calcolo e salvo le medie delle colonne
        self.column_means_ = X_copy.mean()
        
        return self

    def transform(self, X):
        # Creo una copia per non modificare l'input originale
        X_copy = X.copy()
        
        # Sostituisco gli infiniti con NaN
        X_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Riempio i NaN con le medie calcolate durante il fit
        X_filled = X_copy.fillna(self.column_means_)
        
        return X_filled