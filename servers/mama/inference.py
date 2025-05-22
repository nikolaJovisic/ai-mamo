import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from .preprocess_mama import MamaTransform
import einops
import torch.nn as nn
from pydicom import dcmread
from scipy.stats import norm

class SimpleLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return 1 - norm.cdf((self.fc(x).detach().numpy() - 0.64) / 3.45)

transform = MamaTransform()

def get_encoder(weights_path):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.eval()

def get_head(weights_path):
    model = SimpleLinear(768)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.eval()

def infer(encoder, head, img_path):
    img = dcmread(img_path)
    img = img.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

    img = transform(img)
    img = einops.rearrange(img, 'c h w -> 1 c h w')
    embedding = encoder(img)
    return head(embedding).item()

encoder = get_encoder('mama_embed_pretrained_40k_steps_last_dinov2_vit_ckpt.pth')
head = get_head('head_weights.pth')

print('1s')
for entity in os.scandir('/home/nikola/projects/mammography/data/EMBED/small_subset/birads1_dicoms'):
    result = infer(encoder, head, entity.path) 
    print(result)
print('5s')
for entity in os.scandir('/home/nikola/projects/mammography/data/EMBED/small_subset/birads5_dicoms'):
    result = infer(encoder, head, entity.path) 
    print(result)
