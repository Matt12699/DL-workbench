import unittest
import pandas as pd
import numpy as np

from src.data.custom_imputer import CustomImputer

class CustomImputerTest(unittest.TestCase):

    def setUp(self):
        self.imputer = CustomImputer()
        self.df = pd.DataFrame({
            'feature1':[12, 21, np.nan , 32, np.nan, np.inf, -np.inf, 98],
            'feature2':[np.nan, 32, np.nan, np.nan, 43, np.inf, -np.inf, 65],

        })

    # Verifico che nessun valore sia NaN
    def test_noNaN(self):
        processed = self.imputer.fit_transform(self.df)
        print(processed)
        self.assertTrue(not np.any(np.isnan(processed)))

if __name__ == "__main__":
    unittest.main()