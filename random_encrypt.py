import os, sys
import os.path as osp
import numpy as np

from PIL import Image
import gc
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# libraries within this package
from cmd_args import parse_args
from utils.tools import *
from utils.util import generate_code
import datasets
import models

# 数据集整体匿名化
# -------------------- load & set args --------------------
args = parse_args(sys.argv[1])
args.during_training = False

args.old_ckpt_dir = osp.dirname(sys.argv[1])

args.resume = osp.join(args.old_ckpt_dir, args.ckpt_name)

args.gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
args.device = torch.device('cuda:0')
args.batch_size = args.batch_size // 4 * len(args.gpu_ids)
# 数据文件夹有 2 层 或 3层
data_dir_type = args.data_dir_type
# -------------------- dataset & loader --------------------
test_dataset = datasets.__dict__[args.dataset](
    data_dir = args.data_root,
    data_dir_type= data_dir_type,
    # train=False,
    transform=transforms.Compose([
        transforms.Resize((args.imageSize, args.imageSize), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ]),
    # args=args
)
print('test_dataset: ' + str(test_dataset))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)

model_dict = {}

G_input_nc = args.input_nc + args.passwd_length
model_dict['G'] = models.define_G(G_input_nc, args.output_nc,
                                  args.ngf, args.which_model_netG, args.n_downsample_G,
                                  args.normG, args.dropout,
                                  args.init_type, args.init_gain,
                                  args.passwd_length,
                                  use_leaky=args.use_leakyG,
                                  use_resize_conv=args.use_resize_conv,
                                  padding_type=args.padding_type)
model_dict['G_nets'] = [model_dict['G']]

if args.resume:
    if osp.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1

        name = 'G'
        net = model_dict[name]
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(checkpoint['state_dict_' + name])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    gc.collect()
    torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True

model_dict['G'].train()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)


def my_save_img(img, name):
    img_cur = ((img.squeeze(0) + 1.) / 2.)
    save_image(img_cur, name)


with torch.no_grad():
    for i, (img, img_path) in enumerate(tqdm(test_loader)):
    # for i, ((img, label, landmarks, img_path)) in enumerate(tqdm(test_loader)):
    #     if img.size(0) != args.batch_size:
    #         continue

        start = i * args.batch_size
        end = start + args.batch_size

        img_cuda = img.cuda()

        z, dis_target, \
        rand_z, rand_dis_target, \
        inv_z, inv_dis_target, \
        rand_inv_z, rand_inv_dis_target, _, _ = generate_code(args.passwd_length,
                                                             len(img),
                                                              args.device,
                                                              inv=True,
                                                              use_minus_one=args.use_minus_one,
                                                              gen_random_WR=False)
        # print(img_path[0],img[0])
        fake = model_dict['G'](img, z.cpu())

        # batchsize*5, 3, H, W
        if data_dir_type == 1:
            for save_idx in range(len(img)):
                if not os.path.exists(f"{save_dir}/{img_path[save_idx].split('/')[-2]}"):
                    os.makedirs(f"{save_dir}/{img_path[save_idx].split('/')[-2]}")
                my_save_img(fake[save_idx], f"{save_dir}/{img_path[save_idx].split('/')[-2]}/{img_path[save_idx].split('/')[-1]}")
        else:
            for save_idx in range(len(img)):
                my_save_img(fake[save_idx], f"{save_dir}/{img_path[save_idx].split('/')[-3]}/{img_path[save_idx].split('/')[-2]}/{img_path[save_idx].split('/')[-1]}")