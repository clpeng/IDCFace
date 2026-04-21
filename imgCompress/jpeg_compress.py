from glob import glob

import numpy as np
import cv2


def jpeg_compress(image_path, save_path, quality):
    img = cv2.imread(image_path,1)
    save_path = save_path[:save_path.rfind('.')] + '.jpg'
    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY,quality])



# if __name__ == '__main__':
#     img_dir = '/mnt/03a7b3b6-14f4-4a74-958b-71dd6cdd09bc/outputs/fingerprint/CASIA-WebFace-fingerprint_jpeg_0'
#     imgs = glob(img_dir + '/*/*/*/*[!txt]')
#     for img_path in imgs:
#         jpeg_compress(img_path, img_path, quality=0)
