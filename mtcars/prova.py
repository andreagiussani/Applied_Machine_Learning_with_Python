import pandas as pd
import os
import logging

class DatasetCreation:

    logger = logging.getLogger(__name__)

    def __init__(self,logger, filename, col_to_drop=None, categorical_cols = None):
        self.filename = filename
        self.logger = logger
        self.col_to_drop = col_to_drop

    @staticmethod
    def pandas_setting():
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 500)

    @staticmethod
    def query_path(filename):
        root = 'mtcars/'
        path = os.path.join(root, filename + '.csv')
        return path

    def get_name(self):
        return str(self.filename)+'.csv'

    def get_raw_data(self):
        df = pd.read_csv(self.get_name())
        return df

    def split_target(self,df,target_name):
        if self.col_to_drop is None:
            X = df.drop(target_name, axis=1)
            y = df[target_name]
        else:
            X = df.drop(self.col_to_drop, axis=1).drop(target_name,axis=1)
            y = df[target_name]
        return X,y


    def spotting_null_values(self,X):
        summary_stats = {}
        for c in X.columns.values:
            if X[c].dtypes == 'O':
                summary_stats[c] = X[c][X[c].notnull()].mode()
            if X[c].dtypes in ['int64', 'float64']:
                summary_stats[c] = X[c][X[c].notnull()].median()
        for c in list(X):
            X[c].fillna(summary_stats[c], inplace=True)
        return X

    def dummization(self,X,categorical_cols=None):
        df = pd.get_dummies(self.spotting_null_values(X), prefix_sep='_', prefix=categorical_cols,
                                 columns=categorical_cols, drop_first=False)
        return df


