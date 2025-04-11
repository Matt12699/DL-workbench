import logging
import numpy as np

from utilities.logging_config import setup_logging
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew

class CustomBinner(BaseEstimator, TransformerMixin):
    
    def __init__(self, method='Auto', num_bins=10, skewness_theshold = 0.5):
        self.method = method
        self.num_bins = num_bins
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.edges = {}
        self.skewness_theshold = skewness_theshold
        self.warning_printed = False


    def fit(self, X, y=None):

        for column in X.columns:

            # Calcolo l asimmetria per ogni feature per capire quale binning sia il migliore
            skewness = skew(X[column])

            if self.method == 'Uniform':
                # Calcolo i Bin Edges
                self.edges[column] = np.linspace(X[column].min(), X[column].max(), self.num_bins + 1)

            elif self.method == 'Quantile':
                # Riordino la colonna
                X[column] = np.sort(X[column].values)

                # Calcolo i percentili
                percentiles = np.linspace(0, 100, self.num_bins +1 )

                # Calcolo i Bin Edges
                self.edges[column] = np.percentile(X[column], percentiles)

            else: # Questo caso corrisponde ad Auto e forza il comportamento auto in caso di metodo non riconosciuto, si basa sulla skewness
                if self.method != 'Auto' and not self.warning_printed:
                    self.logger.warning("Non-existent method, Auto method is on")
                    self.warning_printed=True
                # Calcolo l asimmetria per capire quale binning sia il migliore
                skewness = skew(X[column])

                if abs(skewness) > self.skewness_theshold:
                    # Utilizzo allora il Quantile Binning
                    # Riordino la colonna
                    X[column] = np.sort(X[column].values)

                    # Calcolo i percentili
                    percentiles = np.linspace(0, 100, self.num_bins +1 )

                    # Calcolo i Bin Edges
                    self.edges[column] = np.percentile(X[column], percentiles)

                else:
                    # Utilizzo l'Uniform Binning
                    # Calcolo i Bin Edges
                    self.edges[column] = np.linspace(X[column].min(), X[column].max(), self.num_bins + 1)

                

        return self
    
    def transform(self, X):

        # Creo una copia dell'input per non lavorarci direttamente
        X_copy = X.copy()

        # Per ogni colonna vado a associare a ogni valore un bin
        for column, edges in self.edges.items():

            X_copy[column] = np.digitize(X[column], edges)  # Assegniamo i bin


        return X_copy