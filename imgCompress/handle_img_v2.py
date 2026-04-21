# 处理图片
import os
from glob import glob

from tqdm import tqdm

from imgCompress.gauss_noise import gauss_noise_compress
from imgCompress.img_crop import crop
from imgCompress.center_crop import center_crop
from imgCompress.gauss_blur import gauss_blur_compress
from imgCompress.jpeg_compress import jpeg_compress


def handle_img(img_dir, type, param, dir_type=1):
    if type == 'or':
        return img_dir


    handle_img_dir = img_dir + f'_{type}_{param}'
    if os.path.exists(handle_img_dir):
        return handle_img_dir
    if dir_type == 1:
        imgs = glob(img_dir + '/fingerprinted_images/*/*[!txt]')
    else:
        imgs = glob(img_dir + '/fingerprinted_images/*/*/*[!txt]')

    print(len(imgs))

    for img_path in tqdm(imgs):
        img_save_path = img_path.replace(img_dir.split('/')[-1], handle_img_dir.split('/')[-1])
        if os.path.exists(img_save_path):
            continue
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        if type == 'jpeg_compress':
            jpeg_compress(img_path, img_save_path, param)
        elif type == 'gauss_noise':
            gauss_noise_compress(img_path, img_save_path, param)
        elif type == 'crop':
            crop(img_path, img_save_path, param)
        elif type == 'gauss_blur':
            gauss_blur_compress(img_path, img_save_path, param)
        elif type == 'center_crop':
            center_crop(img_path, img_save_path, param)

    return handle_img_dir
