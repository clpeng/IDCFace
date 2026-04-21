import argparse
import glob
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import torch
assert insightface.__version__>='0.3'

parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=1, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
parser.add_argument('--model-cpt-root', default='../pretrained_modelspretrained_models', type=str, help='model-cpt-root')
args = parser.parse_args()

app = FaceAnalysis(root=args.model_cpt_root)
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))



def gender_det_single(img_np, verbose=False):
    "dection gender according to face"
    "img_np numpy image, like np.array(image)"
    faces = app.get(img_np)
    gender = 0
    if len(faces) > 0:
        gender = faces[0]['gender']
    if verbose:
        print('M' if gender == 1 else 'F')
    return gender

def gender_det(img_np_array, verbose=False):
    "dection gender according to face"
    "a batch of image array"
    img_size = img_np_array.shape[0]
    gender_arr = torch.empty(img_size)
    for i in range(img_size):
        gender = gender_det_single(img_np_array[i], verbose)
        gender_arr[i] = gender

    return gender_arr
if __name__ == '__main__':
    img = Image.open('../007.jpg')
    img = np.expand_dims(np.array(img), axis=0)
    print(img.shape)
    img_list = np.concatenate((img, img),  axis=0)
    print(img_list.shape)
    gender_det(img_list, True)