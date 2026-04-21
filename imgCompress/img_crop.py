from glob import glob

import numpy as np
import cv2


def crop(image_path, save_path, rate):
    img = cv2.imread(image_path,1)
    h, w = img.shape[:2]
    w = int(w*rate)
    h = int(h*rate)
    x = (img.shape[1] - w)//2
    y = (img.shape[0] - h)//2
    # print(y,y+h, x,x+w)
    img = img[y:y+h, x:x+w]
    save_path = save_path[:save_path.rfind('.')] + '.jpg'
    cv2.imwrite(save_path, img)



if __name__ == '__main__':
    img_path = '../images/006.png'

    crop(img_path, "img_cropped.jpg", rate=.5)
