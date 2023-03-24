import pandas as pd

class DataIngestion:
    """
    This class is used to ingest data into the system before preprocessing.
    """
    def __init__(self,**args):
        """
        This module is used to ingest data into the system before preprocessing.
        """
        self.filename = args.get('filename')
        self.col_to_drop = args.get('col_to_drop')
        self.col_target = args.get('col_target')
        self.X = None
        self.y = None

    def __call__(self, split_features_target:bool=False):
        """
        This function takes the .csv file, and clean from unwanted columns. 
        If split_features_target is set to True the function returns the set of features (explanatory variables)
        and the target variable
        Parameters
        ----------
            split_features_target: bool
                Default value is False, if True return set of features and target variable.
        """
        self.df = pd.read_csv(self.filename,index_col=False)
        self.df = self.df.loc[:, ~self.df.columns.str.match('Unnamed')]
        if split_features_target:
            self.y=self.df[self.col_target]  # This returns a vector containing the target variable
            self.X=self.df.drop(self.col_target,axis=1) if self.col_to_drop is None else self.df.drop(
                    [self.col_to_drop,self.col_target],axis=1
                    )
            return self.X,self.y
        return self.df

    def split_train_test(self, test_size=0.3, random_seed=42):
        """
        This function splits the data into train and test set
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                               test_size=test_size,
                                                                               random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_counts(self, variable_name, title_plot, yticklabels):
        """
        This is more a utility, since it returns the distribution of a variable
        """
        plt.figure(figsize=(8, 5))
        sns.set(font_scale=1.4)
        sns.heatmap(pd.DataFrame(self.df[variable_name].value_counts()), annot=True,
                    fmt='g', cbar=False, cmap='Blues',
                    annot_kws={"size": 20}, yticklabels=yticklabels)
        plt.title(title_plot)
