import unittest
import pandas as pd
import numpy as np

from src.data.frequency_encoder import FrequencyEncoder

class FrequencyEncoderTest(unittest.TestCase):

    def setUp(self):
        self.encoder = FrequencyEncoder(soglia=0.2)
        self.data = pd.DataFrame({
            'Fruit':["Apple", "Banana", "Ananas", "Apple", "Apple", "Banana", "Strawberry", "Watermelon", "Banana"],
            'Car':["Volkswagen", "Ford", "Fiat", "Tesla", "Pagani", "Fiat", "Fiat", "Fiat", "Volkswagen"],
        })
        self.processed = self.encoder.fit_transform(self.data)

    # Verifico che le feature categoriche siano state mappate a valori possibili, i meno frequenti sono stati mappati a 0
    def test_intervallo(self):
        print(self.processed)
        for column in self.processed.columns:
            possible_values = len(self.data[column].unique())
            self.assertTrue((self.processed>=0).all().all() and (self.processed<=possible_values).all().all())

    # Verifico che ogni valore sia stato mappato a un valore unico e che non si crei ambiguità
    def test_valori_unici(self):
        
        # Test per ogni colonna
        for column in self.data.columns:
            
            # Ottengo le categorie sotto la soglia
            value_counts = self.data[column].value_counts(normalize=True) # Associo a ogni valore la sua frequenza
            categories_values_below_threshold = value_counts[value_counts < self.encoder.soglia].index # Ottengo i valori categorici sotto la soglia
            
            # Verifico che tutti i valori categorici sotto la soglia vengano mappati a 0
            for val in categories_values_below_threshold:
                assert (self.processed[column][self.data[column] == val] == 0).all()
            
            # Verifico che i valori delle feature sopra la soglia siano mappati a valori unici
            for valCat, valNum in self.encoder.mapping[column].items():

                if valNum!=0: # I valori categorici mappati a 0 sono già stati controllati
                    assert list(self.encoder.mapping[column].values()).count(valNum) == 1, f"Il valore numerico {valNum} per la categoria {valCat} non è unico."
        

        


if __name__ == "__main__":
    unittest.main()
