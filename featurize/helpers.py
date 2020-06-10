import os
import random
import numpy as np
import gzip
from glob import glob

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from resnet import resnet50

from albumentations import Compose as ACompose
from albumentations.pytorch.transforms import ToTensor as AToTensor
from albumentations.augmentations import transforms as atransforms



NAIP_BAND_STATS = {
    'mean' : np.array([0.38194386, 0.38695849, 0.35312921, 0.45349037])[None,None],
    'std'  : np.array([0.21740159, 0.18325207, 0.15651401, 0.20699527])[None,None],
}

def _naip_normalize(x, **kwargs):
    return (x - NAIP_BAND_STATS['mean']) / NAIP_BAND_STATS['std']

def naip_augmentation_valid():
    return ACompose([
        atransforms.Lambda(name='normalize', image=_naip_normalize),
        AToTensor(),
    ])


class RESISCValidation_geohash(Dataset):
    def __init__(self, path, patch_size=153):

        self.image_paths = glob(os.path.join(path, '*', '*.jpg'))
        self.transform   = naip_augmentation_valid()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        y       = os.path.basename(self.image_paths[idx]).split('_')[0]

        X       = np.array(Image.open(self.image_paths[idx])).astype(np.float) / 255
        X       = np.concatenate((X, np.zeros((X.shape[0], X.shape[1], 1))), axis=2)
        X       = self.transform(image=X)['image']

        return X, y


# HELPERS
def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)

def to_numpy(x):
    return x.detach().cpu().numpy()

def get_pretrained_resnet(model_path):
    model      = resnet50(in_channels=4, num_classes=128)
    dim_mlp    = model.fc.weight.shape[1]
    model.fc   = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    state_dict = torch.load(model_path)['state_dict']
    for k in list(state_dict.keys()):
        if 'encoder_q' not in k:
            del state_dict[k]
    state_dict = {k.replace('module.', '').replace('encoder_q.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = nn.Sequential(*list(model.children()))[:-1]
    return model