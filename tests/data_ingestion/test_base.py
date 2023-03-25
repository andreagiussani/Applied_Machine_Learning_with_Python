import unittest
import pandas as pd

from egeaML.dataingestion import DataIngestion
from tests.data_ingestion.fixture import get_mocked_string_csv


class DataIngestionTestCase(unittest.TestCase):

    def setUp(self):
        self.col_target = 'y'
        self.filename = get_mocked_string_csv()
        self.columns = ['col1', 'col2', 'col3', 'y']
        self.raw_data = DataIngestion(filename=self.filename, col_target=self.col_target)

    def test__load_dataframe(self):
        df = self.raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 4)
        self.assertEqual(df.shape[1], 4)
        self.assertListEqual(list(df), self.columns)
