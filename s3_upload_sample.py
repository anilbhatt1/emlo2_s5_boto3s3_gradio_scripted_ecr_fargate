import boto3
import argparse
import csv

data = [
    ["Name", "Age"],
    ["Alice", 30],
    ["Bob", 25],
    ["Charlie", 35]
]

csv_file_name = "data.csv"

with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

s3 = boto3.client('s3')

bucket_name = 'emlo2s5'
s3_object_name = 'flagged/file.csv'

s3.upload_file(csv_file_name, bucket_name, s3_object_name)
print(f'Successfully uploaded to {bucket_name}/{s3_object_name}')