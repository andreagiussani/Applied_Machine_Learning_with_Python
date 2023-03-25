import unittest
import pandas as pd

from unittest.mock import MagicMock
from egeaML.datareader import FinancialDataReader


class FinancialDataReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.raw_data = FinancialDataReader("AAPL", start_date="2023-01-02", end_date="2023-01-04")
        self.raw_data = MagicMock(return_value=pd.DataFrame(
            [
                [130.279999, 130.899994, 124.169998, 125.070000, 124.879326, 112117500],
                [127.279999, 128, 122.879326, 124.970000, 124.5, 102115400],
            ],
            columns=self.columns
        ))

    def test__load_dataframe(self):
        df = self.raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 2)
        self.assertEqual(df.shape[1], 6)
        self.assertListEqual(list(df), self.columns)
