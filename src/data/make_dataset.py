# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from time import sleep
import os
import gc
import pandas as pd
import numpy as np
from os.path import join as pj
import botocore
import boto3
import pickle as pkl

@click.command()
@click.option('--use-s3', 'use_s3', default=True)
def main(use_s3):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('making final data set from raw data')

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(os.environ.get('BUCKET'))
    raw_dir = pj(project_dir, 'data', os.environ.get('RAW_DIR'))
    processed_dir = pj(project_dir, 'data', os.environ.get('PROCESSED_DIR'))

    filenames = [
        'cc_financial.csv',
        'CharityBase_20200820.csv',
        'cc_class.csv',
        'partb_activities_scraped_2020_08_12_20_18.csv',
        'classification_objects.csv',
        'regno_activities.txt'
    ]

    # download datasets from s3
    if use_s3 is True:
        print('test')
        print('---Using S3---')
        
        for s3_object in bucket.objects.all():
            # Need to split s3_object.key into path and file name, else it will give error file not found.
            path, filename = os.path.split(s3_object.key)
            if filename in filenames:
                bucket.download_file(s3_object.key, pj(raw_dir,filename))
            
        # wait for temp file to change temp name to final (file)name
        sleep(20)
            
    # load scraped part b charities
    partb_activities = pd.read_csv(pj(raw_dir, 'partb_activities_scraped_2020_08_12_20_18.csv'), \
                                    index_col=0).iloc[:,1:]
    partb_activities.dropna(inplace=True)
    partb_activities.drop_duplicates(inplace=True)
    partb_activities['regno'] = partb_activities['regno'].astype(int).astype(str)

    # load scraped non part b charities
    nonpartb_activities = pd.read_csv(pj(raw_dir, 'regno_activities.txt'), delimiter='\t', \
                        lineterminator='\n', header=None, names=['regno', 'activities'], dtype={'regno':str}).dropna()

    activities = pd.concat([partb_activities, nonpartb_activities])
    activities.drop_duplicates('regno', inplace=True)

    # load classifications
    char_classes = pd.read_csv(pj(raw_dir, 'classification_objects.csv'))
    char_classes['regno'] = char_classes['regno'].astype(str)


    # #### Map current classifications to activities data
    classes_map = pd.Series(char_classes['ICNPO_NCVO_category'].values, char_classes['regno'].values).to_dict()
    objects_map = pd.Series(char_classes['objects'].values, char_classes['regno'].values).to_dict()
    name_map = pd.Series(char_classes['nicename'].values, char_classes['regno'].values).to_dict()

    activities['icnpo'] = activities['regno'].map(classes_map)
    activities['name'] = activities['regno'].map(name_map)
    activities['objects'] = activities['regno'].map(objects_map)

    # #### Add income and expenditure to dataframe
    char_financial = pd.read_csv(pj(raw_dir, 'cc_financial.csv'))
    char_financial['regno'] = char_financial['regno'].astype(str)
    char_financial['fyend'] = pd.to_datetime(char_financial['fyend'])

    # reduce to just charities needed
    idx = activities.set_index('regno').index
    char_financial = char_financial.set_index('regno')[char_financial.set_index('regno').index.isin(idx)]\
                                                                                                    .reset_index()
    # add 3 year means
    char_means = {}
    for name, group in char_financial.groupby('regno'):
        df = group[['fyend', 'income']].sort_values('fyend', ascending=False).head(3)
        char_means[name] = df['income'].mean()

    activities['income_3y_mean'] = activities['regno'].map(char_means)

    # prepare data for use
    data = activities.copy()
    data['activities'] = data['activities'].astype(str)
    data.dropna(inplace=True)

    # fix badly formed sentences (words stuck together) - this also removes punctuation
    #data['activities'] = data['activities'].apply(lambda x: ' '.join(wordninja.split(x))) - 

    # add charitybase data
    charitybase = pd.read_csv(pj(raw_dir, 'CharityBase_20200820.csv'))
    charitybase['Charity ID'] = charitybase['Charity ID'].astype(str)
    columns = ['Charity ID', 'LAUA', 'RU', 'EER', 'Funders', 'Trustees']
    charitybase = charitybase[columns]
    data_merged = data.merge(charitybase, left_on='regno', right_on='Charity ID')
    data_merged.dropna(subset=['LAUA', 'RU'], inplace=True)

    # reduce rural/urban categories to just 6
    ru_categories = {
        '(England/Wales) Urban city and town': 'urban_mid',
        '(England/Wales) Urban major conurbation': 'urban_large',
        '(England/Wales) Rural village': 'rural_mid',
        '(England/Wales) Rural town and fringe': 'rural_large',
        '(England/Wales) Rural hamlet and isolated dwellings': 'rural_small',
        '(England/Wales) Urban minor conurbation': 'urban_small',
        '(England/Wales) Rural village in a sparse setting': 'rural_mid',
        '(England/Wales) Rural hamlet and isolated dwellings in a sparse setting': 'rural_small',
        '(England/Wales) Rural town and fringe in a sparse setting': 'rural_large',
        '(England/Wales) Urban city and town in a sparse setting': 'urban_mid',
        '(Scotland) Large Urban Area': 'urban_large',
        '(Scotland) Accessible Rural': 'rural_mid',
        '(Scotland) Other Urban Area': 'urban_small', 
        '(Scotland) Remote Rural': 'rural_small',
        '(Scotland) Very Remote Rural': 'rural_small', 
        '(Scotland) Accessible Small Town': 'urban_mid',
        '(Scotland) Remote Small Town': 'rural_large', 
        '(Scotland) Very Remote Small Town': 'rural_large' 
    }

    data_merged['RU'] = data_merged['RU'].map(ru_categories)

    # add self-classified data
    self_class = pd.read_csv(pj(raw_dir, 'cc_class.csv'), dtype={'self_class': str, 'regno': str})
    self_class.rename(columns={'classtext': 'self_class'}, inplace=True)
    self_class = self_class.groupby('regno')['self_class'].apply(lambda x: ' '.join(map(str, x)))
    data_merged = data_merged.merge(self_class, on='regno')

    # convert income, funders and trustees to log
    data_merged['income_3y_mean'] = data_merged['income_3y_mean'].apply(lambda x: np.log(x) if x > 0 else 0)
    data_merged['Trustees'] = data_merged['Trustees'].apply(lambda x: np.log(x) if x > 0 else 0)
    data_merged['Funders'] = data_merged['Funders'].apply(lambda x: np.log(x) if x > 0 else 0)

    # remove data where charities have0 income in the last 3 years
    data = data_merged[data_merged['income_3y_mean']>0]
    data.reset_index(inplace=True)
    del data_merged

    if use_s3 is True:
        print('---Using S3---')
        ### save file to s3 (and load again)
        print("Saving data...")
        filename = 'data.pkl'
        key = pj('char-class-data', filename)
        pickle_byte_obj = pkl.dumps(data)
        bucket.put_object(Key=key, Body=pickle_byte_obj)
        del pickle_byte_obj
        print('Data saved to s3.')

        # download to local processed folder
        print('Downloading pickle file...')
        try:
            bucket.download_file(key, pj(processed_dir, filename))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    else:
        # save to local processed folder
        filename = 'data.pkl'
        _file = open(pj(processed_dir, filename), 'wb')
        pkl.dump(data, _file)
        print('Saving complete.')

    gc.collect()

    logger.info('final dataset completed')

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
