import torch
import gradio as gr
from torchvision import transforms as T
import argparse

default_output_path = "s3://emlo2s5/outputs"

parser = argparse.ArgumentParser(description='Docker based Inferencing')
parser.add_argument('-f', '--flagged-dir', type=str, default=default_output_path, metavar='N',
                    help='Image path for saving inference outputs')

def demo():
    model_file_name = 'model.script.pt'
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
    )

    demo.launch(server_port=8080, share=True)

if __name__ == "__main__":
    args = parser.parse_args()
    output_path = args.flagged_dir
    demo()