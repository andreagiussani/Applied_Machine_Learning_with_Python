## Helper functions for the book "Applied Machine Learning with Python"

[![PyPi](https://img.shields.io/pypi/v/egeaML.svg)](https://pypi.python.org/pypi/egeaML)
[![Downloads](https://static.pepy.tech/badge/egeaML)](https://pypi.python.org/pypi/egeaML)
[![CI status](https://github.com/andreagiussani/Applied_Machine_Learning_with_Python/workflows/CI/badge.svg)](https://github.com/andreagiussani/Applied_Machine_Learning_with_Python/actions?queryworkflow%3ACI+event%3Apush+branch%3Amain)

<p align="center">
  <img src="cover.jpg" width="428" height="584" title="FrontCover">
</p>


This repository contains the Supplementary Material for the book "Applied Machine Learning with Python", written by Andrea Giussani.
You can find details about the book on the [BUP](https://bup.egeaonline.it) website.  
The book was written with the following versions of some popular libraries:
- [scikit-learn](https://scikit-learn.org/stable/) version 1.2.2
- [pandas](https://pandas.pydata.org) version 1.5.3
- [xgboost](https://xgboost.readthedocs.io/en/latest/#) version 1.7.4
- [gensim](https://radimrehurek.com/gensim/) version 3.8.1
- [matplotlib](https://matplotlib.org) version 3.7.1
- [seaborn](https://seaborn.pydata.org) version 0.9.0

## How to use the EgeaML Library
The book provides a book-specific module called **egeaML**. To add the published package to a project managed with uv, run:
```bash
uv add egeaML
```
To use the version printed with the book, run `uv add "egeaML==0.2.3"` instead.
Once installed you can load a structured label dataset - such as the well-known Boston dataset - 
as a `pandas.DataFrame`, as follows:

```python
from egeaML.datareader import DataReader

raw_data = DataReader(
    filename='https://raw.githubusercontent.com/andreagiussani/datasets/master/egeaML/boston.csv',
    col_target='MEDV'
)
```

The package continues to evolve, so use `egeaML==0.2.3` to match the print version of the book.

## How to develop on the EgeaML
Clone the repository and enter its directory:
```bash
git clone https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
cd Applied_Machine_Learning_with_Python
```
Install the locked dependencies, including the development tools:
```bash
uv sync --group dev
```
This project requires Python 3.13 or later. Run commands inside the environment with `uv run`, for example `uv run pytest`.

### Unittest each method
As a developer, you should unittest your contribution.
To do so, you simply need to create a dedicated folder inside the `tests` subfolder (or possibly extend an existing one),
and test that your method exactly does what you expect. Please look at the following example for inspiration:

```python
import unittest
import os
import pandas as pd

from egeaML.datareader import DataReader


class DataIngestionTestCase(unittest.TestCase):
    URL_STRING_NAME = 'https://raw.githubusercontent.com/andreagiussani/datasets/master/egeaML'
    FILENAME_STRING_NAME = 'boston.csv'

    def setUp(self):
        self.col_target = 'MEDV'
        self.filename = os.path.join(self.URL_STRING_NAME, self.FILENAME_STRING_NAME)
        self.columns = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        self.raw_data = DataReader(filename=self.filename, col_target=self.col_target)

    def test__load_dataframe(self):
        df = self.raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 506)
        self.assertEqual(df.shape[1], 14)
```
The above unittest checks that the output is of type `pandas.DataFrame` and 
verify the expected output satisfies some characteristics.

## Extra Stuff
To use the `egeaML` library in a Jupyter notebook, add Jupyter and the IPython kernel to your environment:
```bash
uv add --dev jupyter ipykernel
uv run python -m ipykernel install --user --name=<YOUR_ENV>
```
Replace `<YOUR_ENV>` with the kernel name you want to see in Jupyter.

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
