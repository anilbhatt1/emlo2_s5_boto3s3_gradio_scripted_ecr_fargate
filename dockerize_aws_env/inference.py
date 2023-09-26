import torch
import boto3
import gradio as gr
from torchvision import transforms as T
import argparse
import CSVLoggerS3, get_pylogger

default_model_path  = "s3://emlo2s5/model.script.pt"
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
    model_file_name = lst[-1]
    bucket_name = lst[2]
    lst2 = lst[3:]
    s3_file_path = '/'.join(lst2)

    # Specify the local path where you want to save the downloaded file
    local_path = './' + model_file_name

    # Use the S3 client to download the file
    try:
        s3.download_file(bucket_name, s3_file_path, local_path)
        print(f'Successfully downloaded {model_file_name} to {local_path}')
    except Exception as e:
        print(f'Error: {str(e)}')

    return model_file_name

def demo(model_file_name, output_path):
    model = torch.jit.load(model_file_name)  

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
        allow_flagging="manual",
        flagging_dir="flagged",
        flagging_callback=CSVLoggerS3(s3_dir=output_path, to_s3=True),
    )

    demo.launch(server_port=8080, share=True)

if __name__ == "__main__":
    args = parser.parse_args()
    s3_bucket_path = args.model   
    output_path = args.flagged_dir
    model_file = download_model(s3_bucket_path)
    demo(model_file, output_path)