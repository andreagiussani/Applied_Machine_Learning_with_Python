from setuptools import setup, find_packages

setup(
    name="egeaML",
    version="1.0.2",
    author="Andrea Giussani",
    author_email="andrea.giussani@unibocconi.it",
    description=("A python library used in support of the Book"
                 "'Applied Machine Learning with Python'"),
    url="https://github.com/andreagiussani/Applied_Machine_Learning_with_Python",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        'pandas==1.5.3', 'scikit-learn==1.2.2',
        'shap==0.41.0', 'catboost==1.1.1',
        'gensim==4.3.1', 'nltk==3.8.1',
        'matplotlib==3.7.1', 'seaborn==0.9.0', 'wget==3.2',
        'imbalanced-learn==0.5.0', 'tensorflow==2.11.0',
        'xgboost==1.7.4',
    ],
    include_package_data=True,
)
