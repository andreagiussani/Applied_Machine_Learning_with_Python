# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold

import re
import nltk
import gensim
import string
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from datetime import datetime
from matplotlib.colors import ListedColormap
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier


class DataIngestion:
    
    def __init__(self,**args):
        self.data = args.get('df')
        self.df = None
        self.col_to_drop = args.get('col_to_drop')
        self.col_target = args.get('col_target')
        self.X = None
        self.y = None

    def load_data(self):
        self.df = pd.read_csv(self.data,index_col=False)
        self.df = self.df.loc[:, ~self.df.columns.str.match('Unnamed')]
        return self.df

    def features(self):
        if self.col_to_drop is None:
            self.X=self.load_data().drop(self.col_target,axis=1)
        else:
            self.X=self.load_data().drop(self.col_to_drop,axis=1).drop(self.col_target,axis=1)
        return self.X

    def target(self):
        self.y=self.load_data()[self.col_target]
        return self.y


class Preprocessing:
    def __init__(self,columns,X):
        self.categorical_cols = columns
        self.X = X
        self.df = None

    def spotting_null_values(self):
        summary_stats = {}
        for c in self.X.columns.values:
            if self.X[c].dtypes == 'O':
                summary_stats[c] = self.X[c][self.X[c].notnull()].mode()
            if self.X[c].dtypes in ['int64', 'float64']:
                summary_stats[c] = self.X[c][self.X[c].notnull()].median()
        for c in list(self.X):
            self.X[c].fillna(summary_stats[c], inplace=True)
        return self.X

    def dummization(self):
        self.df = pd.get_dummies(self.spotting_null_values(), prefix_sep='_', prefix=self.categorical_cols, columns=self.categorical_cols,
                                    drop_first=False)
        return self.df


class model_fitting:

    def __init__(self,n):
        self.n = n

        self.lr = LogisticRegression()
        self.dt = DecisionTreeClassifier()
        self.svc = SVC()

        self.my_dict = dict()
        self.abb_list = list()
        self.names_list = list()
        self.raw_scikit_models =  list()
        self.clfs = None

    """def models_def(self):
        for i in range(0,self.n):
            ask_model=input("Which model do you want to fit? ")
            ask_abb = input("Give a shortcut of it: ")
            self.my_dict[ask_model]=ask_abb
        return self.my_dict
    """
    
    def models_def(self, **kwargs):
        self.my_dict[kwargs['model_one']] = kwargs['abb1']
        self.my_dict[kwargs['model_two']] = kwargs['abb2']
        self.my_dict[kwargs['model_three']] = kwargs['abb3']
        return self.my_dict

    def get_models(self,**kwargs):
        my_dict = kwargs['models_dict']#self.models_def()
        my_list = [kwargs['model_one'],kwargs['model_two'],
                   kwargs['model_three']]
        for name,abb in my_dict.items():
            self.abb_list.append(abb)
            self.names_list.append(name)
        for i in my_list:
            self.raw_scikit_models.append(i+'()')
        scikit_models_list = [self.lr,self.dt,self.svc]
        self.clfs = list(zip(self.names_list,scikit_models_list))
        return self.clfs
    
    def fitting_models(self,models, X_train,y_train,X_test,y_test):
        for name,clf in models:
            clf_ = clf
            clf_.fit(X_train,y_train)
            y_pred = clf_.predict(X_test)
            score = format(accuracy_score(y_test,y_pred), '.4f')
            print("{} : {}".format(name,score))
    

    """def fitting_models(self,X_train,y_train,X_test,y_test):
        for name,clf in self.get_models(
                    model_one='LogisticRegression',
                    model_two='DecisionTreeClassifier',
                    model_three='SVC'):
            clf_ = clf
            clf_.fit(X_train,y_train)
            y_pred = clf_.predict(X_test)
            score = format(accuracy_score(y_test,y_pred), '.4f')
            print("{} : {}".format(name,score))
    """


class plots:
        
    def plot_pca(X):
        
        def draw_vector(v0, v1, ax=None):
            ax = ax or plt.gca()
            arrowprops=dict(facecolor='black',
                            arrowstyle='->',
                        linewidth=2,
                        shrinkA=0, shrinkB=0)
            ax.annotate('', v1, v0, arrowprops=arrowprops)
        
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
        # plot data
        ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
        for length, vector in zip(pca.explained_variance_, 
                                  pca.components_):
            v = vector * 3 * np.sqrt(length)
            draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
        ax[0].axis('equal');
        ax[0].set(xlabel='x', ylabel='y', title='input')
        
        # plot principal components
        X_pca = pca.transform(X)
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
        draw_vector([0, 0], [0, 3], ax=ax[1])
        draw_vector([0, 0], [3, 0], ax=ax[1])
        ax[1].axis('equal')
        ax[1].set(xlabel='component 1', ylabel='component 2',
                  title='principal components',
                  xlim=(-5, 5), ylim=(-3, 3.1))
        
    def plot_loss():
        hy = np.linspace(-3,3,1000)
        plt.plot(hy, np.log(1+np.exp(-hy)), linestyle='-')
        plt.plot(hy, np.maximum(1 - hy, 0), linestyle=':')
        plt.xlim([-3,3])
        plt.ylim([-0.05, 5])
        plt.ylabel("Loss")
        plt.xlabel("Raw Model Output: $h_θ(x) \cdot y$")
        plt.legend(['Logistic', 'Hinge'])


class classification_plots:
    
    def training_class(X,y,test_size=0.3):
        """
        This Function plots a a 2-dim training set,
        and each point is labelled by the class it belongs to.
        The arguments are as follows:
            - X: 2-dim set of features;
            - y: 1-dim target label;
            - test_size: equal to 0.3 by default.
                         It can take any numer between 0 and 1
        """
        plt.style.use('ggplot')
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=test_size,\
                                                         random_state=42)
        df = pd.DataFrame(dict(height=X_train.iloc[:,1], weight=X_train.iloc[:,0],\
                               label=y_train))
        colors = {0:'red', 1:'blue'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='weight', y='height', label=key, \
                       color=colors[key],figsize=(5, 5))
        plt.legend(["Training Class Female", "Training Class Man"],fontsize=10)
        plt.show()
        
    def knn_class(X,y,test_size=0.3):
        """
        This Function fits and a k-Neigh classifier and provides the 
        visualization of the prediction results wrt the target.
        """
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=test_size,random_state=42)
        df = pd.DataFrame(dict(height=X_train.iloc[:,1], weight=X_train.iloc[:,0],
                               label=y_train))
        colors = {0:'red', 1:'blue'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='weight', y='height', label=key, 
                       color=colors[key],figsize=(5, 5))
        clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        df_ = pd.DataFrame(dict(height=X_test.iloc[:,1], weight=X_test.iloc[:,0],
                                label=y_pred))
        colors = {0:'orange', 1:'green'}
        grouped_ = df_.groupby('label')
        for key, group in grouped_:
            group.plot(ax=ax, kind='scatter', x='weight', y='height', label=key, 
                       color=colors[key],figsize=(5, 5))
        plt.xlabel('Weight', fontsize=14)
        plt.ylabel('Height', fontsize=14)
        plt.legend(["Training Female", "Training Man", "Test Pred Female", "Test Pred Man"]
                   ,fontsize=10)
        plt.show()
        
    def plotting_prediction(X_train,X_test,y_train,y_test,nn):
        """
        This function plots the test set points labelled with the predicted value.
        The parameter nn stands for the Number of Neighbors
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(5,5))

        clf = KNeighborsClassifier(n_neighbors=nn).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        colors = ['lime' if i else 'yellow' for i in y_test]
        ps = clf.predict_proba(X_test)[:,1]
        errs = ((ps < 0.5) & y_test) |((ps >= 0.5) & (1-y_test))
        plt.scatter(X_test.weight[errs], X_test.height[errs], facecolors='red', s=150)
        plt.scatter(X_test.weight, X_test.height, 
                    facecolors=colors, edgecolors='k', s=50, alpha=1)
        plt.xlabel('Weight', fontsize=14)
        plt.ylabel('Height', fontsize=14)
        plt.tight_layout()
        
    
    def confusion_matrix(y_test,y_pred, cmap=None):
        """
        This function generates a confusion matrix, which is used as a
        summary to evaluate a Classification predictor.
        The arguments are:
         - y_test: the true labels;
         - y_pred: the predicted labels;
         - cmap: it is the palette used to color the confusion matrix.
                 The available options are:
                  - cmap="YlGnBu"
                  - cmap="Blues"
                  - cmap="BuPu"
                  - cmap="Greens"
         Please refer to the notebook available on the book repo
                    Miscellaneous/setting_CMAP_argument_matplotlib.ipynb
         for further details.
        """
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=cmap)
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
        plt.show()
        
    def confusion_matrix_(y_test,y_pred, cmap,xticklabels,yticklabels):
        """
        This function generates a confusion matrix, which is used as a
        summary to evaluate a Classification predictor.
        The arguments are:
         - y_test: the true labels;
         - y_pred: the predicted labels;
         - cmap: it is the palette used to color the confusion matrix.
                 The available options are:
                  - cmap="YlGnBu"
                  - cmap="Blues"
                  - cmap="BuPu"
                  - cmap="Greens"
         Please refer to the notebook available on the book repo
                    Miscellaneous/setting_CMAP_argument_matplotlib.ipynb
         for further details.
         - xticklabels: description of x-axis label;
         - yticklabels: description of y-axis label
        """
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=cmap,
                   xticklabels=xticklabels, yticklabels=yticklabels)
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
        plt.show()

    def knn_boundaries(X_train,X_test,y_train,y_test,n_neighbors):
        """
        This Function provides the boundaries for a k-Neigh classifier
        """
        # Create color maps
        cmap_bold = ListedColormap(['#FF3333', '#3333FF']) 
        cmap_light = ListedColormap(['#FF9999', '#9999FF'])
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
        y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        
        # Plot also the training points
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Binary classification (k = %i)"
                      % (n_neighbors))
        plt.show()
        
    def scaling_plot():
        import mglearn
        mglearn.plots.plot_scaling()
        
    def plot_hist(data,features_name,target_name):
        data = pd.DataFrame(data, columns=features_name)
        plt.figure(figsize=(20, 15))
        features = list(data)
        for i, col in enumerate(features):
            plt.subplot(3, len(features)/2 , i+1)
            x = data[col]
            plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel(target_name)
            
    def plot_svc_decision_function(model, ax=None, plot_support=True):
        """Plot the decision function for a 2D SVC"""
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    def plot_svc_regularization_effect(X,y,kernel,cmap):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))  
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
        for ax, C in zip(ax, [100.0, 0.1]):
            model = SVC(kernel=kernel, C=C).fit(X, y) 
            ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=cmap)
            classification_plots.plot_svc_decision_function(model, ax) 
            ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
            ax.set_title('C = {0:.1f}'.format(C), size=14)
        plt.show()


class xgboost:
    def fitting(X,y,param_grid,n_jobs,cv):
        clf_xgb = xgb.XGBClassifier(n_jobs=n_jobs, objective="binary:logistic")
        clf = GridSearchCV(clf_xgb, param_grid=param_grid, verbose=1, cv=cv)
        model = clf.fit(X, y)
        return model
    
    def checking_overfitting(X_train,y_train,learning_rate, n_estimators):
        model__ = xgb.XGBClassifier()
        param_grid_ = dict(learning_rate=learning_rate, 
                   n_estimators=n_estimators)
        grid_search = GridSearchCV(model__, param_grid_, 
                           scoring="neg_log_loss", 
                           n_jobs=-1, 
                           cv=10)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best Log Score: %f using %s" % (grid_result.best_score_, 
                             grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        scores = np.array(means).reshape(len(learning_rate),
                                 len(n_estimators))
        for i, value in enumerate(learning_rate):
            plt.plot(n_estimators, scores[i],label='learning_rate: ' + str(value))
        plt.legend()
        plt.xlabel('n_estimators')
        plt.ylabel('Log Loss')
        plt.savefig('n_estimators_vs_learning_rate.png')


class NLP:
    def __init__(self,text):
        self.text = text
    
    def clean_text(self):
        new_string = []
        for word in gensim.utils.simple_preprocess(self.text):
            if word not in gensim.parsing.preprocessing.STOPWORDS and len(word) > 2:
                stem_ = SnowballStemmer('english')
                lemma = WordNetLemmatizer()
                new = stem_.stem(lemma.lemmatize(word, pos='v'))
                new_string.append(new)
        return new_string


class neural_network:
    
    def plot_decision_boundary(func, X, y, figsize=(6, 6)):
        amin, bmin = X.min(axis=0) - 0.1
        amax, bmax = X.max(axis=0) + 0.1
        hticks = np.linspace(amin, amax, 101)
        vticks = np.linspace(bmin, bmax, 101)
    
        aa, bb = np.meshgrid(hticks, vticks)
        ab = np.c_[aa.ravel(), bb.ravel()]
        c = func(ab)
        cc = c.reshape(aa.shape)

        cm = 'RdBu'
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig, ax = plt.subplots(figsize=figsize)
        contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)

        ax_c = fig.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
        plt.xlim(amin, amax)
        plt.ylim(bmin, bmax)

    def plot_multiclass_decision_boundary(model, X, y):
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                             np.linspace(y_min, y_max, 101))
        cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
        Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()], 
                                  verbose=0)
        Z = Z.reshape(xx.shape)
        fig = plt.figure(figsize=(8, 8))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
    def plot_data(X, y, figsize=None):
        if not figsize:
            figsize = (8, 6)
        plt.figure(figsize=figsize)
        plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
        plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1, marker="^")
        plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
        plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
        plt.legend()
    
    def plot_loss_accuracy(history):
        historydf = pd.DataFrame(history.history, index=history.epoch)
        plt.figure(figsize=(10, 6))
        historydf.plot(ylim=(0, max(1, historydf.values.max())), 
                       style=['+-','.-'] )
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    
    def plot_loss(history):
        historydf = pd.DataFrame(history.history, index=history.epoch)
        plt.figure(figsize=(8, 6))
        historydf.plot(ylim=(0, historydf.values.max()))
        plt.title('Loss: %.3f' % history.history['loss'][-1])
        
    def plot_confusion_matrix(model, y, y_pred):
        plt.figure(figsize=(10, 8))
        mat = pd.DataFrame(confusion_matrix(y, y_pred))
        sns.heatmap(mat, cbar=False, annot=True, fmt='d',
                    cmap='YlGnBu', alpha=0.8, vmin=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
    
    def plot_compare_histories(history_list, name_list,
                               plot_accuracy=True):
        dflist = []
        for history in history_list:
            h = {key: val for key, val in 
                 history.history.items() if not key.startswith('val_')}
            dflist.append(pd.DataFrame(h, index=history.epoch))
    
        historydf = pd.concat(dflist, axis=1)
    
        metrics = dflist[0].columns
        idx = pd.MultiIndex.from_product([name_list, metrics], 
                                         names=['model', 'metric'])
        historydf.columns = idx
        
        plt.figure(figsize=(6, 8))
    
        ax = plt.subplot(211)
        historydf.xs('loss', axis=1, 
                     level='metric').plot(ylim=(0,1), ax=ax)
        plt.title("Loss")
        
        if plot_accuracy:
            ax = plt.subplot(212)
            historydf.xs('acc', axis=1,
                         level='metric').plot(ylim=(0,1), ax=ax)
            plt.title("Accuracy")
            plt.xlabel("Epochs")
    
        plt.tight_layout()
        
    def make_sine_wave():
        c = 3
        num = 2400
        step = num/(c*4)
        np.random.seed(0)
        x0 = np.linspace(-c*np.pi, c*np.pi, num)
        x1 = np.sin(x0)
        noise = np.random.normal(0, 0.1, num) + 0.1
        noise = np.sign(x1) * np.abs(noise)
        x1  = x1 + noise
        x0 = x0 + (np.asarray(range(num)) / step) * 0.3
        X = np.column_stack((x0, x1))
        y = np.asarray([int((i/step)%2==1) for i in range(len(x0))])
        return X, y
    
    def make_multiclass(N=500, D=2, K=3):
        
        """
        N: # points per class
        D: #dimensionality
        K: # of classes  
        """
        
        np.random.seed(0)
        X = np.zeros((N*K, D))
        y = np.zeros(N*K)
        for j in range(K):
            ix = range(N*j, N*(j+1))
            # radius
            r = np.linspace(0.0,1,N)
            # theta
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            y[ix] = j
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdYlBu', alpha=0.8)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        return X, y
