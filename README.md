## Helper functions for the book "Applied Machine Learning with Python"

[![PyPi](https://img.shields.io/pypi/v/egeaML.svg)](https://pypi.python.org/pypi/egeaML)
[![Downloads](https://static.pepy.tech/badge/egeaML)](https://pypi.python.org/pypi/egeaML)

<p align="center">
  <img src="cover.jpg" width="428" height="584" title="FrontCover">
</p>


This repository contains the Supplementary Material for the book "Applied Machine Learning with Python", written by Andrea Giussani.
You can find details about the book on the [BUP](https://bup.egeaonline.it) website.  
The books was written with the following specific versions of some popular libraries:
- [scikit-learn](https://scikit-learn.org/stable/) version 1.2.2
- [pandas](https://pandas.pydata.org) version 1.5.3
- [xgboost](https://xgboost.readthedocs.io/en/latest/#) version 1.7.4
- [gensim](https://radimrehurek.com/gensim/) version 3.8.1
- [matplotlib](https://matplotlib.org) version 3.7.1
- [seaborn](https://seaborn.pydata.org) version 0.9.0

## How to use the EgeaML Library
The book provides a book-specific module, called **egeaML**. <br>
Be sure you have created a virtualenv. Then run 
```bash
pip install egeaML
```
Once installed you can load a structured label dataset - such as the well-known Boston dataset - 
as a `pandas.DataFrame`, as follows:
```python
from egeaML.egeaML import DataIngestion
raw_data = DataIngestion(
    df='https://raw.githubusercontent.com/andreagiussani/Applied_Machine_Learning_with_Python/master/data/boston.csv', 
    col_target='MEDV'
)
```

## How to develop on the EgeaML
Please, clone on your local machine this repo, as follows:
```bash
git clone https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
To install it into your local env, I recommend to create a virtualenv where you add the necessary requirements, running this command from your favourite terminal emulator:
```bash
pip install -r requirements.txt
pip install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```

If, instead, you use the Anaconda system:
```bash
conda install --file requirements.txt
conda install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
If you have Python3 already installed in your local environment, you can run:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```

### Unittest each method
As a developer, you should unittest your contribution.
To do so, you simply need to create a dedicated folder inside the `tests` subfolder (or possibly extend an existing one),
and test that your method exactly does what you expect. Please look at the following example to tke inspiration:
```python
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
        df = self.raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 506)
        self.assertEqual(df.shape[1], 14)
```
The above unittest checks that the output is of type `pandas.DataFrame` and 
verify the expected output satisfies some characteristics.

## Extra Stuff
If you wish to use the `egeaML` library on a Jupyter notebook, you firstly need to install the jupyter library,
and then running the following command
```bash
pip install jupyter
python3 -m ipykernel install --user --name=<YOUR_ENV>
```
where the name is the name you have assigned to your local environment. 
You are now ready to use all the feature of this helper!

## Submitting Errata
If you have errata for the book, please submit them via the [BUP](https://bup.egeaonline.it) website. In case of possible mistakes within the book-specific module, you can submit a fixed-version as a pull-request in this repository.

## How to Cite this Book

```tex
@book{giussani2020,
	TITLE="Applied Machine Learning with Python",
	AUTHOR="Andrea Giussani",
	YEAR="2020",
	PUBLISHER="Bocconi University Press"
}
```
