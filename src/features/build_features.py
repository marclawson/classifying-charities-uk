# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline, make_union, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import tempfile
import joblib
import botocore
import boto3
from os.path import join as pj
import os
import pickle as pkl
from src.features.custom_transformers import (
    FeatureExtractorText,
    FeatureExtractorOHE,
    FeatureExtractorNumber,
    CustomImputer
)



@click.command()
@click.option('--use-s3', 'use_s3', default=True)
def main(use_s3):
    """ Builds features for modelling."""

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(os.environ.get('BUCKET'))
    processed_dir = pj(project_dir, 'data', os.environ.get('PROCESSED_DIR'))

    activities_pipe = make_pipeline(
        FeatureExtractorText('activities'),
        CountVectorizer(),
        StandardScaler()
    )

    objects_pipe = make_pipeline(
        FeatureExtractorText('objects'),
        CountVectorizer(),
        StandardScaler()
    )

    income_pipe = make_pipeline(
        FeatureExtractorNumber('income_3y_mean'),
        StandardScaler()
    )

    title_pipe = make_pipeline(
        FeatureExtractorText('name'),
        CountVectorizer(),
        StandardScaler()
    )

    region_pipe = make_pipeline(
        FeatureExtractorOHE('EER'),
        OneHotEncoder(drop='first'),
        StandardScaler()
    )

    ru_pipe = make_pipeline(
        FeatureExtractorOHE('RU'),
        OneHotEncoder(drop='first'),
        StandardScaler()
    )

    funders_pipe = make_pipeline(
        FeatureExtractorNumber('Funders'),
        CustomImputer(),
        StandardScaler()
    )

    trustees_pipe = make_pipeline(
        FeatureExtractorNumber('Trustees'),
        CustomImputer(),
        StandardScaler()
    )

    selfclass_pipe = make_pipeline(
        FeatureExtractorText('self_class'),
        CountVectorizer(),
        StandardScaler()
    )

    feature_union = make_union(activities_pipe, objects_pipe, income_pipe, title_pipe, region_pipe, 
                    ru_pipe, trustees_pipe, selfclass_pipe)

    
    # save feature_union
    filename = 'feature_union.jlib'
    if use_s3:
        with tempfile.TemporaryFile() as fp:
            key = pj('char-class-data', filename)
            joblib.dump(feature_union, fp)
            fp.seek(0)
            bucket.put_object(Key=key, Body=fp.read())
        # download to local processed folder
        try:
            bucket.download_file(key, pj(processed_dir, filename))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    else:
        _file = open(pj(processed_dir, filename), 'wb')
        joblib.dump(feature_union, _file)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()