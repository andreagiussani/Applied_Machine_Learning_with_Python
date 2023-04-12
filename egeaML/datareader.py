import datetime
from calendar import monthrange
from typing import Union

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

import logging
logging.basicConfig(level=logging.INFO)


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
    Parameters
    ----------
    crypto_name : string
        Cryptocurrency to download
    start_date: datetime, str
        Download start date string (YYYY-MM-DD) or _datetime.
    end_date: datetime, str
        Download end date string (YYYY-MM-DD) or _datetime.
    timeframe : str
        Valid timeframes: 1s,1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d

    Examples
    --------
    Using datetime objects:

    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2022, 12, 31)

    crypto = CryptoDataReader('BTCUSDT', start_date, end_date, '1d')
    data = crypto.get_data()

    Using date as string:

    start_date = '2022-06-30'
    end_date = '2023-03-31'

    crypto = CryptoDataReader('ADAUSDT', start_date, end_date, '1h')
    data = crypto.get_data()
    """

    def __init__(self, crypto_name, start_date, end_date, timeframe):
        self.crypto_name = crypto_name.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self._validation_input()

    def _validation_input(self):

        valid_timeframe = ['1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if self.timeframe not in valid_timeframe:
            raise ValueError(f'Timeframe := {self.timeframe} must be in ({", ".join(valid_timeframe)})')

        if isinstance(self.start_date, str) and isinstance(self.end_date, str):
            self.start_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d').date()
            self.end_date = datetime.datetime.strptime(self.end_date, '%Y-%m-%d').date()

        if self.start_date > self.end_date:
            raise ValueError(f'The end date must be greater than the start date.')

        filepath = 'https://raw.githubusercontent.com/binance/binance-public-data/master/data/symbols.txt'
        tickers = pd.read_csv(filepath, header=None)
        if self.crypto_name not in tickers.values:
            raise ValueError(f'{self.crypto_name} is not a valid ticker. Check the available tickers at {filepath}.')

    @staticmethod
    def _check_connection() -> bool:
        with requests.head('http://www.google.com') as response:
            if response.status_code == 404:
                return False
            else:
                return True

    def _get_url(self, date: datetime, type: str) -> str:
        """ Create the url from where download data """
        year, month, day = date.year, date.strftime('%m'), date.strftime('%d')

        URL = f"https://data.binance.vision/data/spot/" \
              f"{type}/klines/{self.crypto_name}/{self.timeframe}/{self.crypto_name}-{self.timeframe}-{year}-{month}"
        return URL + ".zip" if type == 'monthly' else URL + f"-{day}.zip"

    def _download_data(self, date: datetime, type: str) -> Union[pd.DataFrame, bool]:

        url = self._get_url(date, type=type)
        with requests.get(url) as response:

            if response.status_code == 404:
                return False

            else:
                zipfile = ZipFile(BytesIO(response.content))
                with zipfile.open(zipfile.namelist()[0]) as file_in:
                    download = pd.read_csv(file_in,
                                           usecols=[0, 1, 2, 3, 4, 5, 8, 9],
                                           header=None,
                                           names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'Buy_volume'])
                return download

    @staticmethod
    def last_day_of_month(date: datetime) -> datetime:
        return date.replace(day=monthrange(date.year, date.month)[1])

    def _get_dates_to_download(self) -> list:
        adjusted_end_date = self.last_day_of_month(self.end_date)
        dates_monthly = [(date, 'monthly') for date in pd.date_range(self.start_date, adjusted_end_date, freq='M')]
        dates_daily = [(date, 'daily') for date in pd.date_range(datetime.date.today().replace(day=1), datetime.date.today(), freq='D')
                       if adjusted_end_date >= datetime.date.today()]

        return dates_monthly + dates_daily

    def get_data(self) -> pd.DataFrame:

        if not self._check_connection():
            raise OSError('No connection available')

        data = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'Buy_volume'])

        with ThreadPoolExecutor(max_workers=10) as exe:

            dates_to_download = self._get_dates_to_download()
            futures = [exe.submit(self._download_data, date, type) for date, type in dates_to_download]

            for future in as_completed(futures):
                output = future.result()
                if isinstance(output, pd.DataFrame):
                    data = pd.concat([data, output], axis=0, join='inner')

        data.drop_duplicates(subset=['Time'], inplace=True)
        data.Time = pd.to_datetime(data.Time, unit='ms')
        data.set_index(keys='Time', inplace=True)
        data.sort_index(inplace=True)
        data = data.loc[self.start_date:datetime.datetime.combine(self.end_date, datetime.time(23, 59, 59))]

        if data.index.min().date() != self.start_date or data.index.max().date() != self.end_date:
            logging.info(f'Data for {self.crypto_name} is only available '
                         f'from {data.index.min().date()} to {data.index.max().date()}')

        return data
