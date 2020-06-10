#!/usr/bin/env python

"""
    featurize.py
"""

import os
import gzip
import json
import geohash
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from helpers import set_seeds, to_numpy, get_pretrained_resnet, RESISCValidation_geohash


out_path = '/home/ebarnett/DiscriminativeActiveLearning/featurize/resisc/'

# --
# Load model
model = get_pretrained_resnet('/home/bjohnson/projects/moco/models/naip/checkpoint_0030.pth.tar')
model = model.cuda()
model = model.eval()

# --
# Data loaders
dataset = RESISCValidation_geohash(path='/raid/users/ebarnett/NWPU-RESISC45/')
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)

# --
# Featurize
labs = []
embs = []

for idx, (xx, lab) in tqdm(enumerate(loader), total=len(loader)):
    with torch.no_grad():
        xx = xx.cuda()

        out = to_numpy(model(xx))
        embs.append(out.reshape(xx.shape[0], -1))
        labs.append(lab)

labs = np.hstack(labs)
embs = np.row_stack(embs)

np.save(os.path.join(out_path, 'labs'), labs)
np.save(os.path.join(out_path, 'embs'), embs)