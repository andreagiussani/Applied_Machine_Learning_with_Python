import datetime

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import requests
from io import BytesIO
from zipfile import ZipFile

from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split

from egeaML.constants import (
    UNNAMED_COLNAME,
    FILENAME_CONSTANT,
    COL_TARGET_COLNAME,
    COL_TO_DROP_CONSTANT,
)

import warnings


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning


class DataReader:
    """
    This class is used to ingest data into the system before preprocessing.
    """

    def __init__(self, **args):
        """
        This module is used to ingest data into the system before preprocessing.
        """
        self.filename = args.get(FILENAME_CONSTANT)
        self.col_to_drop = args.get(COL_TO_DROP_CONSTANT)
        self.col_target = args.get(COL_TARGET_COLNAME)
        self.X = None
        self.y = None

    def __call__(self, split_features_target: bool = False):
        """
        This function takes the .csv file, and clean from unwanted columns.
        If split_features_target is set to True the function returns the set of features (explanatory variables)
        and the target variable
        Parameters
        ----------
            split_features_target: bool
                Default value is False, if True return set of features and target variable.
        """
        df = pd.read_csv(self.filename, index_col=False)
        df = df.loc[:, ~df.columns.str.match(UNNAMED_COLNAME)]
        if split_features_target:
            self.y = df[self.col_target]  # This returns a vector containing the target variable
            self.X = df.drop(self.col_target, axis=1) if self.col_to_drop is None else \
                df.drop([self.col_to_drop, self.col_target], axis=1)
            return self.X, self.y
        return df

    def split_train_test(self, test_size=0.3, random_seed=42):
        """
        This function splits the data into train and test set
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_seed
        )
        return X_train, X_test, y_train, y_test

    def plot_column_distribution(self, variable_name, title_plot, yticklabels):
        """
        This is a graphical utility, since it returns the distribution of a variable
        """
        plt.figure(figsize=(8, 5))
        sns.set(font_scale=1.4)
        sns.heatmap(
            pd.DataFrame(self.df[variable_name].value_counts()),
            annot=True,
            fmt='g', cbar=False, cmap='Blues',
            annot_kws={"size": 20},
            yticklabels=yticklabels
        )
        plt.title(title_plot)


class FinancialDataReader:
    # TODO: to be improved

    def __init__(self, stock_name, start_date, end_date):
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self._validation_input()

    def _validation_input(self):
        if type(self.stock_name) is not str:
            raise ValueError('The stock name must be a string')
        if self.start_date > self.end_date:
            raise ValueError('The end date must be greater than the start date.')

    def __call__(self):
        df = yf.download(self.stock_name, start=self.start_date, end=self.end_date)
        return df


class CryptoDataReader:
    """
    example:
    sd = datetime.date(2022, 1, 1)
    ed = datetime.date(2022, 12, 31)
    crypto = CryptoDataReader('BTCUSDT', sd, ed, '1d')
    data = crypto.get_data()
    """

    def __init__(self, crypto_name, start_date, end_date, timeframe):
        self.crypto_name = crypto_name
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self._validation_input()

    def _validation_input(self):
        # TODO:
        #       (1) check if symbol is in binance list
        #       (2) check if timeframe is valid
        #       (3) check if date are datetime object or valid string
        pass

    def _get_url(self, date, type):
        """ Create the url from where download data """
        year, month, day = date.year, date.strftime('%m'), date.strftime('%d')

        if type == 'monthly':
            URL = "https://data.binance.vision/data/spot/monthly/klines/"
            return URL + f"{self.crypto_name}/{self.timeframe}/{self.crypto_name}-{self.timeframe}-{year}-{month}.zip"

        elif type == 'daily':
            URL = "https://data.binance.vision/data/spot/daily/klines/"
            return URL + f"{self.crypto_name}/{self.timeframe}/{self.crypto_name}-{self.timeframe}-{year}-{month}-{day}.zip"

    def _download_data(self, date, type):

        url = self._get_url(date, type=type)
        with requests.get(url) as response:
            if response.status_code == 404:
                # TODO:
                #       (1) check if it is a connection error or there's no such a file
                pass

            else:
                zipfile = ZipFile(BytesIO(response.content))
                with zipfile.open(zipfile.namelist()[0]) as file_in:
                    download = pd.read_csv(file_in, header=None)
                return download

    @staticmethod
    def last_day_of_month(date):
        if date.month == 12:
            return 31
        else:
            return (date.replace(month=date.month + 1, day=1) - datetime.timedelta(days=1)).day

    def get_data(self):

        adjusted_end_date = datetime.date(self.end_date.year,
                                          self.end_date.month,
                                          self.last_day_of_month(self.end_date))

        data = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(self._download_data, date, 'monthly')
                       for date in pd.date_range(self.start_date, adjusted_end_date, freq='M')]

            for future in as_completed(futures):
                output = future.result()
                if isinstance(output, pd.DataFrame):
                    data = pd.concat([data, output])

        data.drop(columns=[6, 7, 9, 10, 11], inplace=True)
        data.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        data.Time = pd.to_datetime(data.Time, unit='ms')
        data.set_index(keys='Time', inplace=True)
        data.sort_index(inplace=True)
        data = data.loc[self.start_date:self.end_date]

        if data.index.min().date() != self.start_date or data.index.max().date() != self.end_date:
            warnings.warn(f'Download Warning: Data for {self.crypto_name} is only available '
                          f'from {data.index.min().date()} to {data.index.max().date()}')

        return data
