from torchvision.transforms import Resize, CenterCrop
from torchvision import transforms as T
import cv2
import numpy as np
from PIL import Image


def center_crop(image_path, save_path, rate=1.0):

    image = Image.open(image_path)
    w,h = image.size[:2]
    img_size = int(rate * min(w, h))
    transforms = T.Compose([CenterCrop(img_size)])
    image = transforms(image)
    image.save(save_path)


if __name__ == '__main__':
    img_path = '../images/000001.png'
    center_crop(img_path,"img_center_crop.jpg", rate=1.5)