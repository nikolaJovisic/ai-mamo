import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from preprocess_mama import MamaTransform
import einops
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

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
    img = Image.open(img_path)
    img = transform(img)
    img = einops.rearrange(img, 'c h w -> 1 c h w')
    embedding = encoder(img)
    return head(embedding).item()

encoder = get_encoder('encoder_weights.pth')
head = get_head('head_weights.pth')

result = infer(encoder, head, 'sample.png')
print(result)


