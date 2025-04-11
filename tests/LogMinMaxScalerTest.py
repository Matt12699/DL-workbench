import unittest
import pandas as pd
import numpy as np

from src.data.log_min_max_scaler import LogMinMaxScaler

class TestLogMinMaxScaler(unittest.TestCase):

    # Creo lo scaler e un DataFrame di prova
    def setUp(self):
        self.scaler = LogMinMaxScaler()
        self.data = pd.DataFrame({
            'feature1' : [3 , 10000 , 32, 459999, 345000000, 211111, 232, 3543],
            'feature2' : [32, 45, 6, 55, 67, 99, 12, 14]
        })

    def test_intervalloGiusto(self):
        
        # Applico la trasformazione
        processed = self.scaler.fit_transform(self.data)
        
        # Utilizzo una piccola tolleranza 
        tolleranza = 1e-10
        print(processed)
        # Verifico l'intervallo con la tolleranza
        self.assertTrue( np.all(processed >= -1 - tolleranza) and  np.all(processed <= 1 + tolleranza), "Ci sono dei valori fuori intervallo")

if __name__ == "__main__":
    unittest.main()