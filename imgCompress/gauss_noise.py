import numpy as np
from PIL import Image
import skimage
from matplotlib import pyplot as plt

#
# def gauss_noise_compress(image_path, save_path, var, mean=0, amplitude=1.0):
#     '''
#         添加⾼高斯噪声
#         image:原始图像
#         mean : 均值
#         var : ⽅方差,越⼤大，噪声越⼤大
#     '''
#     image = Image.open(image_path)
#     img = np.array(image)
#     h, w, c = img.shape
#     N = amplitude * np.random.normal(loc=mean, scale=var, size=(h, w, 1))
#     N = np.repeat(N, c, axis=2)
#     img = N + img
#     img[img > 255] = 255  # 避免有值超过255而反转
#     img = Image.fromarray(img.astype('uint8')).convert('RGB')
#     img.save(save_path)
#     return img


def gauss_noise_compress(img_path, save_path, var, mean=0):
    img = Image.open(img_path)
    img = np.array(img)
    re = skimage.util.random_noise(img,var=var,mean=mean)
    plt.imsave(save_path, re, format='jpeg')


if __name__ == '__main__':
    img_path = '../images/006.png'
    # gauss_noise_compress(img_path,save_path='sd_2.jpg',var=0.05)
    # img = np.array(Image.open(img_path))
    # print(img.shape)
