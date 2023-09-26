import torch
import boto3
import gradio as gr
from torchvision import transforms as T
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_model_path  = "s3://emlo2s5/model_weights/model.script.pt"
default_output_path = "s3://emlo2s5/outputs"

parser = argparse.ArgumentParser(description='Docker based Inferencing')
parser.add_argument('-m', '--model', type=str, default=default_model_path, metavar='N',
                    help='S3 model to be used for inferencing')

parser.add_argument('-f', '--flagged-dir', type=str, default=default_output_path, metavar='N',
                    help='Image path for saving inference outputs')

def download_model(s3_bucket_path):
    # Initialize an S3 client
    s3 = boto3.client('s3')
    logger.info(f'Inside download model s3_bucket_path : {s3_bucket_path}')
    # Specify the bucket name and file name
    lst = s3_bucket_path.split('/')
    model_file_name = lst[-1]
    bucket_name = lst[2]
    lst2 = lst[3:]
    s3_file_path = '/'.join(lst2)

    # Specify the local path where you want to save the downloaded file
    local_path = './' + model_file_name
    
    logger.info(f'bucket_name is {bucket_name}')
    logger.info(f's3_file_path is {s3_file_path}')
    logger.info(f'local_path is {local_path}')  
    logger.info(f'model_file_name is {model_file_name}') 

    # Use the S3 client to download the file
    try:
        s3.download_file(bucket_name, s3_file_path, local_path)
        logger.info(f'Successfully downloaded {model_file_name} to {local_path}')
        print(f'Successfully downloaded {model_file_name} to {local_path}')
    except Exception as e:
        logger.info(f'S3 download Error: {str(e)}')
        print(f'S3 download Error: {str(e)}')

    current_directory = os.getcwd()
    logger.info(f'Inside download model -Current dir is {current_directory}')
    directory_contents = os.listdir(current_directory)
    for item in directory_contents:
        logger.info(f'--**item is {item}')
        
    return model_file_name

def demo(model_file_name):
    current_directory = os.getcwd()
    logger.info(f'Current dir is {current_directory}')
    directory_contents = os.listdir(current_directory)
    for item in directory_contents:
        logger.info(f'--item is {item}')
    model_file_path = './' + model_file_name
    logger.info(f'Model_file_path is {model_file_path}')
    model = torch.jit.load(model_file_path)  

    class_name = ['airplane', 'car','birds', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def recognize_cifar_image(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        preds = model.forward_jit(image)        
        preds = preds[0].tolist()
        return {str(class_name[i]): preds[i] for i in range(10)}

    im = gr.Image(shape=(32, 32),type="pil")

    demo = gr.Interface(
        fn=recognize_cifar_image,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
    )

    demo.launch(server_port=8080, share=True)

if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(f'Args : {args}')
    s3_bucket_path = args.model   
    logger.info(f' Args s3_bucket_path  : {s3_bucket_path}')
    output_path = args.flagged_dir
    logger.info(f' Args output_path  : {output_path}')
    model_file = download_model(s3_bucket_path)
    logger.info(f' After download model_file  : {model_file}')
    demo(model_file)