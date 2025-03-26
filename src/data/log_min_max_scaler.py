import logging
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

# SRC_TO_DST_SECOND_BYTES        44477 valori NULLI

class LogMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min_max_scaler = MinMaxScaler(feature_range=self.feature_range)
    
    # Apprendimento dei parametri
    # y è obbligatoria anche se non la utiliziamo
    def fit(self, X, y=None):
        X = self._check_values(X)  # Controllo valori prima della trasformazione
        # Applico la trasformazione logaritmica 
        X_log = np.log1p(X)  # Log1p è log(1+x) per evitare problemi con zeri
        
        # Applico il MinMaxScaler
        self.min_max_scaler.fit(X_log)
        return self

    # Trasformazione vera e propria
    def transform(self, X):
        X = self._check_values(X)  # Controllo valori prima della trasformazione
        # Applico la trasformazione logaritmica
        X_log = np.log1p(X)  # Log1p è log(1+x) per evitare problemi con zeri
        
        # Applico il MinMaxScaler
        X_scaled = self.min_max_scaler.transform(X_log)
        return X_scaled
    
    def _check_values(self, X):
        # Controlla NaN e valori negativi prima del log 
        if np.any(X < -1):  
            raise ValueError("Il dataset contiene valori < -1, impossibili da trasformare con log1p.")
        if np.any(np.isnan(X)):
            print(X)
            raise ValueError("Il dataset contiene NaN, rimuovili o imputali prima della trasformazione: ", np.isnan(X).sum())
        return X