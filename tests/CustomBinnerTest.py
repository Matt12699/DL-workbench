import unittest
import pandas as pd
import numpy as np

from src.data.custom_binner import CustomBinner

class CustomBinnerTest(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1' : [3 , 10000 , 32, 459999, 345000000, 211111, 232, 3543],
            'feature2' : [32, 45, 6, 55, 67, 99, 12, 14]
        })
    
    def test_discretizzazione_delle_feature_Uniform(self):

        encoder = CustomBinner('Uniform', 3)
        processed = encoder.fit_transform(self.data)
        self.assertTrue(np.all(processed >= 1) and  np.all(processed <= 4), "Values out of bounds")
    
    def test_discretizzazione_delle_feature_Quantile(self):

        encoder = CustomBinner('Quantile', 3)
        processed = encoder.fit_transform(self.data)
        self.assertTrue(np.all(processed >= 1) and  np.all(processed <= 4), "Values out of bounds")

    def test_discretizzazione_delle_feature_Auto(self):

        encoder = CustomBinner('Auto', 3)
        processed = encoder.fit_transform(self.data)
        self.assertTrue(np.all(processed >= 1) and  np.all(processed <= 4), "Values out of bounds")

    def test_discretizzazione_delle_feature_No_Method(self):

        encoder = CustomBinner('Pippo', 3)
        processed = encoder.fit_transform(self.data)
        self.assertTrue(np.all(processed >= 1) and  np.all(processed <= 4), "Values out of bounds")

    
if __name__=="__main__":
    unittest.main()