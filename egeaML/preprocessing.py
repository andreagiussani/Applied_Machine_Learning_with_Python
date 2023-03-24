import pandas as pd

class Preprocessing:
    """
    This class is aimed at facilitiating preprocessing.
    It is made of two main objects: detecting nulls,
    and dealing with categorical_cols.
    """
    def __init__(self, columns, X):
        self.categorical_cols = columns
        self.X = X
        self.df = None

    def simple_imputer(self):
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
        self.df = pd.get_dummies(self.simple_imputer(), prefix_sep='_', prefix=self.categorical_cols, columns=self.categorical_cols,
                                 drop_first=False)
        return self.df