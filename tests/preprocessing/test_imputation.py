import unittest
import pandas as pd
import numpy as np

from egeaML.egeaML import Preprocessing


class DataIngestionTestCase(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(
            {
                'col1': [1, 6, np.nan, 5],
                'col2': [100, np.nan, np.nan, 30],
                'col3': ['iphone', 'iphone', np.nan, 'pixel']
            }
        )
        self.transformer = Preprocessing(columns=None, X=self.X)

    def test__impute_null_values(self):
        df = self.transformer.simple_imputer()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df[~df['col2'].isna()].shape[0], 4)
        self.assertEqual(df.loc[2, 'col3'], 'iphone')
        self.assertEqual(df.loc[2, 'col1'], 5)
