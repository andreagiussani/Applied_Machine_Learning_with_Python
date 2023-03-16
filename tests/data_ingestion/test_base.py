import unittest
import os
import pandas as pd

from egeaML.egeaML import DataIngestion


class DataIngestionTestCase(unittest.TestCase):

    URL_STRING_NAME = 'https://raw.githubusercontent.com/andreagiussani/Applied_Machine_Learning_with_Python/master/data'
    FILENAME_STRING_NAME = 'boston.csv'

    def setUp(self):
        self.col_target = 'MEDV'
        self.filename = os.path.join(self.URL_STRING_NAME, self.FILENAME_STRING_NAME)
        self.columns = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        self.raw_data = DataIngestion(df=self.filename, col_target=self.col_target)

    def test__load_dataframe(self):
        df = self.raw_data.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 506)
        self.assertEqual(df.shape[1], 14)
        self.assertListEqual(list(df), self.columns)
