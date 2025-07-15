#!/usr/bin/env python3
import boto3
import botocore
import joblib
from io import BytesIO
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
from src.features.custom_transformers import (
    FeatureExtractorText,
    FeatureExtractorOHE,
    FeatureExtractorNumber,
    CustomImputer
)

def get_model_from_s3(bucket_name, filename):
    s3 = boto3.client('s3')
    try:
        with BytesIO() as fp:
            s3.download_fileobj(bucket_name, filename, fp)
            fp.seek(0)
            model = joblib.load(fp)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
            return None
        else:
            raise
    return model

def list_s3_objects(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        print(f"Objects in s3://{bucket_name}/{prefix}:")
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print(f"No objects found in s3://{bucket_name}/{prefix}")

@click.command()
@click.argument('text')
def main(text):
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    filename = os.path.join(os.environ.get('MODELS_DIR'), "default_cart_bc_2020_08_26_22_06.jlib")
    bucket_name = os.environ.get('BUCKET')
    list_s3_objects(bucket_name, 'char_class_data/models/')
    print(f"Loading model from S3 bucket: {bucket_name}, file: {filename}")
    model = get_model_from_s3(bucket_name, filename)
    if model is None:
        print("Model could not be loaded. Exiting.")
        return
    print("Model loaded successfully.")

    prediction = model.predict([text])
    print(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    main()