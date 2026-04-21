from glob import glob

import numpy as np
import cv2


def gauss_blur_compress(image_path, save_path, kernel_size):
    img = cv2.imread(image_path,1)
    img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    cv2.imwrite(save_path, img)



if __name__ == '__main__':
    img_path = '../images/000001.png'
    gauss_blur_compress(img_path, 'gauss_blur.jpg', kernel_size=21)
