import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, soglia):
        self.mapping={}
        self.soglia = soglia

    def fit(self, X, y=None):
        
        for column in X.columns:
            
            # Creo una mappa che associa a ogni valore che appare nella colonna la sua frequenza
            freq_map = X[column].value_counts(normalize = True)

            # Seleziono solo i valori che hanno una frequenza maggiore o uguale alla soglia
            top_categories = freq_map[freq_map >= self.soglia]

            # Ordino le categorie per frequenza decrescente
            sorted_categories = top_categories.sort_values(ascending=False)
            
            # Mappo i valori che passano la soglia con un indice unico
            # Assegno un valore numerico per ogni categoria, creando un dizionario formato da valore categorico: valore intero
            mapping={}
            for idx, cat in enumerate(sorted_categories.index):
                mapping[cat] = idx + 1
            self.mapping[column] = mapping


        return self

    def transform(self, X):
        # Non lavoro sull'input originale
        X_copy = X.copy()

        for column in X_copy.columns:  
            # Mappo ogni valore nella colonna usando la mappa
            X_copy[column] = X_copy[column].map(self.mapping[column]).fillna(0)

        return X_copy