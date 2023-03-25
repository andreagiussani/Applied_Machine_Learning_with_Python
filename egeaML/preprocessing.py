import pandas as pd

from typing import List


class Preprocessing:
    """
    This class is aimed at facilitating preprocessing.
    It is made of two main objects: detecting nulls,
    and dealing with categorical_cols.
    """
    def __init__(self, columns: List, X: pd.DataFrame):
        self.categorical_cols = columns
        self.X = X
        self.df = None

    def simple_imputer(self) -> pd.DataFrame:
        """
        This function replaces null values in the input DataFrame with mode for object columns
        and median for numeric columns.
        """
        summary_stats = self.X.select_dtypes(include=['object']).mode().to_dict(orient='records')[0]
        summary_stats.update(self.X.select_dtypes(exclude=['object']).median().to_dict())
        self.X.fillna(value=summary_stats, inplace=True)
        return self.X

    def dummization(self):
        """
        This function performs dummization for categorical_cols
        """
        #TODO: use sklearn ColumnTransformer instead

        return pd.get_dummies(
            self.simple_imputer(),
            prefix_sep='_',
            prefix=self.categorical_cols,
            columns=self.categorical_cols,
            drop_first=False
        )
