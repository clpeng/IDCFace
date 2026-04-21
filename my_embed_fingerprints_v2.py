import argparse
import os
import glob
import pickle

import PIL

# 为了避免包内存,图像立即存储
parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true",
                    help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true",
    help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument(
    "--data_dir_type",
    type=int,
    default=1,
    help="type of data dir",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, data_dir_type, transform=None):
        self.data_dir = data_dir
        self.data_dir_type = data_dir_type
        if data_dir_type == 1:
            self.filenames = glob.glob(data_dir+"/*/*.*[!txt]")
        else:
            self.filenames = glob.glob(data_dir + "/*/*/*.*[!txt]")

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

        print("imgs size", len(self.filenames))
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_dir = filename[:filename.rfind('/')]
        id_path = img_dir + '/id.txt'

        with open(id_path, 'rb') as f:
            # 按保存变量的顺序加载变量
            id = pickle.load(f)
            # print(id)

        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, id

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, args.data_dir_type, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, 1)
    fingerprints = fingerprints.view(1, FINGERPRINT_SIZE).expand(BATCH_SIZE, FINGERPRINT_SIZE)
    fingerprints = fingerprints.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0
    cnt = 0
    for images, id in tqdm(dataloader):
        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, FINGERPRINT_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)
        id = id.to(device)

        fingerprinted_images = HideNet(id[: images.size(0)], images)
        cur_fingerprinted_images = fingerprinted_images.detach().cpu()
        all_fingerprints.append(id[: images.size(0)].detach().cpu())

        # save images
        dataset_size = len(dataset.filenames)
        for idx in range(len(cur_fingerprinted_images)):
            image = cur_fingerprinted_images[idx]
            image_idx = (cnt * BATCH_SIZE) + idx
            if image_idx >= dataset_size:
                print("idx error", image_idx)
                break
            filepath = dataset.filenames[image_idx]
            if args.data_dir_type == 1:
                dir_name = filepath.split('/')[-2]
            else:
                dir_name = filepath.split('/')[-3] + '/' + filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
            filename = filename[:filename.rfind('.')] + ".png"
            # print(filename)
            if not os.path.exists(os.path.join(args.output_dir, "fingerprinted_images", f"{dir_name}")):
                os.makedirs(os.path.join(args.output_dir, "fingerprinted_images", f"{dir_name}"))

            save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{dir_name}/{filename}"),
                       padding=0)



        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[
                                                                                     : images.size(0)]).float().mean(
                dim=1).sum().item()
        cnt += 1
    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
        os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprints)):
        fingerprint = all_fingerprints[idx]
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        filepath = dataset.filenames[idx]
        dir_name = filepath.split('/')[-2]
        filename = filepath.split('/')[-1]
        f.write(f"{dir_name}/{filename} {fingerprint_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        save_image(torch.abs(images - fingerprinted_images)[:49],
                   os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)


def main():
    load_data()
    load_models()

    embed_fingerprints()


if __name__ == "__main__":
    main()
