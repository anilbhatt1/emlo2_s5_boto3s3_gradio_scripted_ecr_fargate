import boto3
import argparse

# Initialize an S3 client
s3 = boto3.client('s3')

bucket_name = 'emlo2s5'
s3_file_path = 'model_weights/requirements.txt'
local_path = './req.txt'

# Use the S3 client to download the file
try:
    s3.download_file(bucket_name, s3_file_path, local_path)
    print(f'Successfully downloaded to {local_path}')
except Exception as e:
    print(f'Error: {str(e)}')