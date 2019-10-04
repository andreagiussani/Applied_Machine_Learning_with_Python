## Helper functions for the book "Applied Machine Learning with Python"

<p align="center">
  <img src="cover.png" width="408" height="454" title="FrontCover">
</p>


This repository contains the Supplementary Material for the book "Applied Machine Learning with Python", written by Andrea Giussani.
You can find details about the book on the [BUP](https://bup.egeaonline.it) website.  
The books was written with the following specific versions of some popular libraries:
- [scikit-learn](https://scikit-learn.org/stable/) version 0.20.4
- [pandas](https://pandas.pydata.org) version 0.23.1
- [numpy](https://numpy.org) version 1.16.4
- [xgboost](https://xgboost.readthedocs.io/en/latest/#) version 0.82
- [nltk](https://www.nltk.org) version 3.3
- [matplotlib](https://matplotlib.org) version 3.1.1
- [seaborn](https://seaborn.pydata.org) version 0.9.0

## EgeaML
The book provides a book-specific module, called **egeaML**. To install it into your local environment, use the command

```
pip install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
or using Anaconda:
```
conda install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
If you have Python3 already installed in your local environment:

```
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
To use it inside your Python3 environment, you should initialise the class as follows:
```
import egeaML as eml
```
or alternatively
```
from egeaML import *
```

## Submitting Errata
If you have errata for the book, please submit them via the [BUP](https://bup.egeaonline.it) website. In case of possible mistakes within the book-specific module, you can submit a fixed-version as a pull-request in this repository.

## How to Cite this Book

```tex
@BOOK{giussani2019,
	TITLE="Applied MAchine Learning with Python",
	AUTHOR="Andrea Giussani",
	YEAR="2019",
	PUBLISHER="Egea"
}
```
