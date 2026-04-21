
import PIL


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)


import torch





def load_decoder(decoder_path, image_resolution, cuda, device):
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(image_resolution, 3, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)





def extract_fingerprints(img_path):
    image = image = PIL.Image.open(img_path)
    fingerprint = RevealNet(image)
    fingerprint = (fingerprint > 0).long()
    fingerprint_str = (str, fingerprint.cpu().long().numpy().tolist())



