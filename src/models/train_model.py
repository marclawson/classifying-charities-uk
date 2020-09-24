# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os
from os.path import join as pj
import pickle as pkl
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from model_params import parameters
import botocore
import boto3
import joblib
from datetime import datetime
import tempfile
import gc
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
import sys
sys.path.append('../..')
from src.features.build_features import FeatureExtractorText, FeatureExtractorOHE, FeatureExtractorNumber, CustomImputer


@click.command()
@click.option('--use-s3', 'use_s3', default=True)
@click.option('--estimator', 'estimator', required=True, type=click.Choice(list(parameters()[0].keys())))
@click.option('--test-size', 'test_size', required=True, type=click.Choice([str(i) for i in range(2, 100)]))
@click.option('--custom-stopwords', 'custom_stopwords', default=[], type=list)
def main(estimator, test_size, custom_stopwords, use_s3):
    """ Trains model on data."""

    #print(list(parameters()[0].keys()))

    processed_dir = pj(project_dir, 'data', os.environ.get('PROCESSED_DIR'))

    # load data
    filename = 'data.pkl'
    _file = open(pj(processed_dir, filename), 'rb')
    data = pkl.load(_file)

    # load features
    filename = 'feature_union.jlib'
    _file = open(pj(processed_dir, filename), 'rb')
    feature_union = joblib.load(_file)

    # split data in train-test sets
    data_copy = data.copy()
    y = data_copy.pop('icnpo')
    X = data_copy

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=int(test_size)/100,
                                                stratify=y,
                                                random_state=1)
   

    # set up estimator and gridsearch
    params, job_args = parameters()
    job_args = job_args['kwargs']
    clf_name = estimator
    estimator = params[clf_name]['estimator']
    param_grid = params[clf_name]['param_grid']

    # create model pipeline
    pipe = Pipeline([
            ('featureunion', feature_union),
            ('clf', estimator)
    ])

    # set class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # stop words for model
    stoplist = stopwords.words('english')
    if custom_stopwords is True:
        extra_words = custom_stopwords
    else:
        extra_words = ['information', 'recorded', 'activities', 'people', 'support', 'raising',
            'provide', 'community', 'public', 'kingdom', 'united', 'charity', 'funds', 'raise']
    stoplist.extend(extra_words)

    # set up fixed arguments for gridsearch
    fixed_params = {
        'clf__max_iter': 100000,
        'clf__class_weight': class_weight_dict,
        'featureunion__pipeline-1__countvectorizer__lowercase': True,
        'featureunion__pipeline-1__countvectorizer__strip_accents': 'unicode',
        'featureunion__pipeline-1__countvectorizer__stop_words': stoplist,
        'featureunion__pipeline-2__countvectorizer__lowercase': True,
        'featureunion__pipeline-2__countvectorizer__strip_accents': 'unicode',
        'featureunion__pipeline-2__countvectorizer__stop_words': stoplist,
        'featureunion__pipeline-4__countvectorizer__lowercase': True,
        'featureunion__pipeline-4__countvectorizer__strip_accents': 'unicode',
        'featureunion__pipeline-4__countvectorizer__stop_words': stoplist,
        'featureunion__pipeline-1__standardscaler__with_mean': False,
        'featureunion__pipeline-2__standardscaler__with_mean': False,
        'featureunion__pipeline-4__standardscaler__with_mean': False,
        'featureunion__pipeline-5__standardscaler__with_mean': False,
        'featureunion__pipeline-6__standardscaler__with_mean': False,
        'featureunion__pipeline-8__standardscaler__with_mean': False
    }

    # remove fixed params that are not needed for various models
    if 'cart' in clf_name:
        del fixed_params['clf__max_iter']
        del fixed_params['clf__class_weight']
    elif 'knn' in clf_name:
        del fixed_params['clf__max_iter']
        del fixed_params['clf__class_weight']

    # set model parameters and fit
    pipe.set_params(**fixed_params)
    searchcv = GridSearchCV(pipe, param_grid=param_grid, **job_args)
    try:
        if __name__ == '__main__':
            searchcv.fit(X_train, y_train)
    except Exception as e:
        raise
        print(e)

    # save model
    now = datetime.now().strftime('%Y_%m_%d_%H_%M')
    filename = f'searchcv_{clf_name}_{now}.jlib'
    if use_s3 is True:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(os.environ.get('BUCKET'))
        with tempfile.TemporaryFile() as fp:
            key = pj('capstone-data', filename)
            joblib.dump(searchcv, fp)
            fp.seek(0)
            bucket.put_object(Key=key, Body=fp.read())
    else:
        models_dir = pj(project_dir, os.environ.get('MODELS_DIR'))
        _file = open(pj(models_dir, filename), 'wb')
        joblib.dump(searchcv, _file)
    
    # delete model from memory as it might be large
    del searchcv
    gc.collect()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()