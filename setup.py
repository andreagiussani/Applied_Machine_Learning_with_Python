from setuptools import setup, find_packages

setup(
    name="egeaML",
    version="1.0.2",
    author="Andrea Giussani",
    description=("A python library used in support of the Book"
                 "'Applied Machine Learning with Python'"),
    url="https://github.com/andreagiussani/Applied_Machine_Learning_with_Python",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        'pandas==3.1.1', 'scikit-learn==1.9.1',
        'shap==0.54.1', 'catboost==1.2.10',
        'gensim==4.4.0', 'nltk==3.10.4',
        'matplotlib==3.11.2', 'seaborn==0.13.2', 'wget==3.2',
        'imbalanced-learn==0.14.2', 'tensorflow==2.21.0',
        'xgboost==3.4.1', 'yfinance==1.8.0', 'requests==2.34.2'
    ],
    include_package_data=True,
)
