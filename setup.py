from setuptools import setup, find_packages

setup(
    name="egeaML",
    version="0.1.0",
    author="Andrea Giussani",
    author_email="andrea.giussani@unibocconi.it",
    description=("A python library used in support of the Book"
                 "'Applied Machine Learning with Python'"),
    url="https://github.com/andreagiussani/Applied_Machine_Learning_with_Python",
    license="BSD",
    packages=["egeaML"],
    install_requires=[
        'pandas==1.5.3', 'scikit-learn==1.2.2', 'xgboost==0.82',
        'shap==0.41.0', 'catboost==1.1.1',
        'gensim==3.8.1', 'nltk==3.4.5',
        'matplotlib==3.1.0', 'seaborn==0.9.0', 'wget==3.2',
        'imbalanced-learn==0.5.0', 'tensorflow==2.11.0',
    ],
    include_package_data=True,
    classifiers=("Programming Language :: Python :: 3"),
)
