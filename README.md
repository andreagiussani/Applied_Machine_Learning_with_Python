## Helper functions for the book "Applied Machine Learning with Python"

<p align="center">
  <img src="cover.jpg" width="428" height="584" title="FrontCover">
</p>


This repository contains the Supplementary Material for the book "Applied Machine Learning with Python", written by Andrea Giussani.
You can find details about the book on the [BUP](https://bup.egeaonline.it) website.  
The books was written with the following specific versions of some popular libraries:
- [scikit-learn](https://scikit-learn.org/stable/) version 0.21.3
- [pandas](https://pandas.pydata.org) version 0.23.1
- [numpy](https://numpy.org) version 1.16.4
- [xgboost](https://xgboost.readthedocs.io/en/latest/#) version 0.82
- [nltk](https://www.nltk.org) version 3.3
- [gensim](https://radimrehurek.com/gensim/) version 3.8.1
- [matplotlib](https://matplotlib.org) version 3.1.0
- [seaborn](https://seaborn.pydata.org) version 0.9.0

## EgeaML
The book provides a book-specific module, called **egeaML**. <br>
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
```
conda install --file requirements.txt
conda install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
If you have Python3 already installed in your local environment, you can run:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/andreagiussani/Applied_Machine_Learning_with_Python.git
```
To use it inside your Python3 environment, you should initialise the class as follows:
```python
import egeaML.egeaML as eml
```
or alternatively
```python
from egeaML.egeaML import *
from egeaML.egeaML import DataIngestion
```

If you wish to use the `egeaML` library on a Jupyter notebook, you firstly need to install the jupyter using
```bash
pip install jupyter
python3 -m ipykernel install --user --name=<YOUR_ENV>
```
and then you are ready to use all the feature of this helper.

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
