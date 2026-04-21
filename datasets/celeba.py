# from random_encrypt

import os.path as osp
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import glob

__all__ = ['CELEBA']


class CELEBA(data.Dataset):
    def __init__(self, data_dir, data_dir_type, transform=None):
        self.data_dir = data_dir
        self.data_dir_type = data_dir_type
        self.root = data_dir
        # print(data_dir,data_dir_type)
        if data_dir_type == 1:
            self.file_paths = glob.glob(data_dir + "/*/*.*[!txt]")
        else:
            self.file_paths = glob.glob(data_dir + "/*/*/*.*[!txt]")

        # self.filenames = sorted(self.filenames)
        # help = []
        # for path in self.filenames:
        #     if args.data_dir_type == 1:
        #         dir_name = path.split('/')[-2]
        #     else:
        #         dir_name = path.split('/')[-3] + '/' + path.split('/')[-2]
        #
        #     filename = path.split('/')[-1]
        #     filename = filename.split('.')[0] + ".png"
        #
        #     save_path = os.path.join(args.output_dir, "fingerprinted_images", f"{dir_name}/{filename}")
        #     if not os.path.exists(save_path):
        #         help.append(path)
        #
        # self.filenames = help
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):

        img_path = self.file_paths[index]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of imgs: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
