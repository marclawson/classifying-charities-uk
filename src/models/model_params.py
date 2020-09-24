#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np


def parameters():

    model_params_dict = {}
    kwargs_dict = {}
    
    model_params_dict['logit'] = {
        'estimator': LogisticRegression(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__penalty': ['l1','l2'],
                'clf__solver': ['saga'],                                      
                'clf__C': np.logspace(-1, 4, 10)
        }
    }

    model_params_dict['knn'] = {
        'estimator': KNeighborsClassifier(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__n_neighbors': list(range(1, 50, 4))
        }
    }

    model_params_dict['cart'] = {
        'estimator': DecisionTreeClassifier(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
                'clf__min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
                'clf__max_depth': np.linspace(1, 32, 32, endpoint=True)
            }
    }

    model_params_dict['cart_bag'] = {
        'estimator': RandomForestClassifier(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__n_estimators': [10, 500, 500, 1000, 5000],
                'clf__max_features': ['auto', 'sqrt', 'log2'],
                'clf__max_samples': [0.1, 0.5, 1.1]
            }
    }

    model_params_dict['cart_boost'] = {
        'estimator': AdaBoostClassifier(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__base_estimator': [DecisionTreeClassifier(max_depth=3)],
                'clf__n_estimators': [10, 500, 500, 1000, 5000]
            }
    }

    model_params_dict['svm'] = {
        'estimator': SVC(),
        'param_grid': {
                'featureunion__pipeline-1__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-1__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-2__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-2__countvectorizer__ngram_range': [(1,3), (2,2)],
                'featureunion__pipeline-4__countvectorizer__max_features': [1000, 5000],
                'featureunion__pipeline-4__countvectorizer__ngram_range': [(1,3), (2,2)],
                'clf__C': np.logspace(-2, 2, 11),
                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'clf__gamma': ['scale'],
                'clf__degree': [2, 3]
            }
    }

    kwargs_dict['kwargs'] = {
        'cv': 5,                                                            
        'n_jobs': -1,                                                        
        'verbose': 10,                                                   
        'scoring': 'accuracy'
    }

    return model_params_dict, kwargs_dict