import torch
import PIL
import timm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
from argparse import ArgumentParser
# To do: 
# 1- Load model in timm package
# 2- Function Two infer an image


class Inferece():

    def __init__(self, image, 
                 model, weight,
                 num_cls):
        self.image = self.convert_image(image)
        
        self.model = timm.create_model(model_name=model, 
                                       checkpoint_path=weight,
                                       num_classes=num_cls, 
                                       in_chans=3)

    @property
    def infer(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        self.imgage = self.image.to(device)
        return self.model(self.image)
    
    @staticmethod
    def convert_image(image):
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image)
        elif isinstance(image, str):
            img = Image.open(image)
            t = transforms.PILToTensor()
            image_tensor = t(img)

        elif isinstance(image, PIL.Image.Image):
            t = transforms.PILToTensor()
            image_tensor = t(image)
        elif isinstance(img, torch.Tensor):
            image_tensor = image
        else:
            raise Exception('Image format is Unknown! must be (string, torch.tensor, np.ndarray or PIL.Image.Image)')

        print(image_tensor.shape)

        image_tensor = image_tensor.unsqueeze(0)
        resize_transform = transforms.Resize((224, 224), antialias=True)
        image_tensor = resize_transform(image_tensor)

        return image_tensor




if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--model', help='model name', default='resnet18.a1_in1k')
    parser.add_argument('--weight', default='./weights/best.pt')
    parser.add_argument('--cls', type=int, help='Number of classes', default=84)
    parser.add_argument('--image', help='test image: np.ndarray, str, torch.Tensor or PIL.Image.Image')
    args = parser.parse_args()



    prediction = Inferece(args.image,
                    model=args.model,
                    weight=args.weight,
                    num_cls=args.cls,
                    ).infer
    
