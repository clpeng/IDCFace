# 处理图片
import os
from glob import glob
from gauss_noise import gauss_noise_compress
from img_crop import crop
from center_crop import center_crop
from gauss_blur import gauss_blur_compress
from jpeg_compress import jpeg_compress

type = 'crop'
# type = 'center_crop'

# 裁剪比例
center_corp_rate = 1.2

# 高斯模糊kernel_size
kernel_size = 73

# 裁剪比例
corp_rate = 0.5

# jepg 压缩质量
quality = 70

# 高斯噪声 0.01
var = 5

# 数据集文件夹
key = 'facescrub_images_encrypt_fingerprint'
img_dir = f'/mnt/03a7b3b6-14f4-4a74-958b-71dd6cdd09bc/outputs/' \
          f'fingerprint_megaface/{key}'



imgs = glob(img_dir + '/*/*/*[!txt]')
print(len(imgs))

if type == 'jpeg':
    save_dir_name = f'{key}_{type}_{quality}'

    for img_path in imgs:
        img_save_path = img_path.replace(key, save_dir_name)
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        jpeg_compress(img_path, img_save_path, quality=quality)
elif type == 'gauss_noise':
    save_dir_name = f'{key}_{type}_{str(var)}'

    for img_path in imgs:
        img_save_path = img_path.replace(key, save_dir_name)
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        gauss_noise_compress(img_path, img_save_path, var)

elif type == 'crop':
    save_dir_name = f'{key}_{type}_{str(corp_rate)}'

    for img_path in imgs:
        img_save_path = img_path.replace(key, save_dir_name)
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        crop(img_path, img_save_path, corp_rate)


elif type == 'blur':
    save_dir_name = f'{key}_{type}_{str(kernel_size)}'

    for img_path in imgs:
        img_save_path = img_path.replace(key, save_dir_name)
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        gauss_blur_compress(img_path, img_save_path, kernel_size)

elif type == 'center_crop':
    save_dir_name = f'{key}_{type}_{str(center_corp_rate)}'

    for img_path in imgs:
        img_save_path = img_path.replace(key, save_dir_name)
        img_save_dir = img_save_path[:img_save_path.rfind('/')]
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        center_crop(img_path, img_save_path, center_corp_rate)
