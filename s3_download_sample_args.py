import boto3
import argparse

default_model_path = "s3://emlo2s5/model_weights/model.script.pt"
default_output_path = "s3://emlo2s5/outputs"

parser = argparse.ArgumentParser(description='Docker based Inferencing')
parser.add_argument('-m', '--model', type=str, default=default_model_path, metavar='N',
                    help='S3 model to be used for inferencing')

parser.add_argument('-f', '--flagged-dir', type=str, default=default_output_path, metavar='N',
                    help='Image path for saving inference outputs')

def download_model(s3_bucket_path):
    # Initialize an S3 client
    s3 = boto3.client('s3')

    # Specify the bucket name and file name
    lst = s3_bucket_path.split('/')
    model_file_name = lst[-1]  # For s3://emlo2s5/model_weights/model.script.pt it will be 'model.script.pt``
    print('Model file Name is: ', model_file_name, 'lst is :', lst)
    bucket_name = lst[2] # Here emlo2s5
    print('Bucket Name is: ', bucket_name)
    lst2 = lst[3:]  # Here 'model_weights/model.script.pt'
    s3_file_path = '/'.join(lst2)
    print('S3 File Path is: ', s3_file_path)

    # Specify the local path where you want to save the downloaded file - './model.script.pt'
    local_path = './' + model_file_name

    # Use the S3 client to download the file
    try:
        s3.download_file(bucket_name, s3_file_path, local_path)
        print(f'Successfully downloaded {model_file_name} to {local_path}')
    except Exception as e:
        print(f'Error: {str(e)}')

    return model_file_name

if __name__ == "__main__":
    args = parser.parse_args()
    s3_bucket_path = args.model   
    print('s3_bucket_path path is: ', s3_bucket_path)
    output_path = args.flagged_dir
    print('Output path is: ', output_path)  
    model_file = download_model(s3_bucket_path)
    print('Model file is: ', model_file)
