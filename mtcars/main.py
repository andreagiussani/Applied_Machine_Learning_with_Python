
import logging
import sys

from prova import DatasetCreation

if __name__ == '__main__':

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    log.info("Start Creating Dataset")

    cl = DatasetCreation(log,'mtcars')

    cl.pandas_setting()

    df = cl.get_raw_data()

    df['performance'] = df['mpg'].apply(lambda x: 'Low' if x < 19 else 'Medium' if x < 22 else 'High')

    df['category'] = df['hp'].apply(lambda x: 'Normal' if x < 150 else 'Sporty')

    X, y = cl.split_target(df,'category')

    log.info("We impute the nulls with median and mode for numerical and categorical variables, respectively.")

    X = cl.spotting_null_values(X)

    log.info("Pivoting the categorical variables - aka Dummization Phase.")

    df_new = cl.dummization(X,categorical_cols=['performance'])

    print(df_new.columns)

    log.info("End Creating Dataset")

    print(df.head())


