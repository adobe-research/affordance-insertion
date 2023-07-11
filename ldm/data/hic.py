import os
import sys
import wandb
import pickle
import einops
import numpy as np
from pathlib import Path

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data_hic.humans_dataset_ldm import HumansDataset

# sample loaders for testing

"""
The data-loader is controlled via configs. A sample config is here:

target: ldm.data.hic.HiC
      params:
        path: '/scratch/pip_data/'
        split: 'train'
        resolution: 256
        flip_p: 0.5
        config:
          mode: 'train'
          masks:
            - 30
            - 15
            - 20
            - 15
            - 20
          dilation: 20
          dilation_mode: 'random'
          augment: True
          zero_person: False
          cfg_mode: 2
          aug_type: 'video'

Base params:
path: Full path to the prepared data LMDBs root directory
split: Split of data to use either 'train' or 'test'. By default,
        50K videos are reserved as 'test'. You can adjust this param.
resolution: Resolution to load the dataset in
flip_p: Flip probability. Both scene image and reference person
        are flipped with this prob.

Config params:
mode: Can be one of 'train', 'test', 'overfit', 'swap', 'halves'
        This param loads the specified split in the specified mode. 
        'train' mode: Samples two random frames from a video,
                    prepare samples for training (has cfg and so on).
        'test' mode: For the test split, try to insert person from 
                    last frame into the first frame.
        'swap' mode: For the test split, take two different videos
                    and insert person from first frame of video #1
                    into scene from first frame of video #2.
        'halves' mode: Instead of masking full person in scene image,
                    only mask one half (chosen at random). Useful for
                    testing partial body completions.
        'overfit' mode: loads 1 video many times, for debugging or
                    test overfitting.
masks: Five integers (that sum to 100) that specify the percentage
        ratio of different types of masks in following order:
        ['bbox', 'pmask', 'scribble', 'smallbox', 'largebox']
        'bbox': Bounding box around the person
        'pmask': Segmentation mask around the person
        'scribble': Random scribbles (but within the person bbox)
        'smallbox': Randomly sampled smaller bbox around the person
        'largebox': Randomly sampled larger bbox around the person
dilation: Specify the number of pixels to dilate the mask by.
dilation_mode: 'random' or 'fixed'. 'random' dilates by a random
        value sampled between [0, 'dilation'] and 'fixed' dilates by
        fixed amount equal to 'dilation' param.
augment: Boolean var to turn on data augmentation. Note data augment
        is turned on only if 'mode' is 'train' and 'augment' is True.
data_type: Either 'video' or 'image'. Use 'video' by default. 'image'
        takes scene & person from the same image, useful for data
        ablations.
zero_person: Zero out the person. Useful for generating person
        hallcuination samples.
clip (optional): To specify a different CLIP model for refer person
        feature extraction. Useful for ablations only.
"""

class HiCRandBoxTest(HumansDataset):
    def __init__(self, **kwargs):
        config = {'mode': 'swap', \
            'masks': [100, 0, 0, 0, 0], \
            'dilation': 10, \
            'dilation_mode': 'fixed',
            'data_type': 'video',
            'zero_person': False,
            'augment': False
        }
        super().__init__(split="test", path="./data/", \
            flip_p=0.0, config=config, **kwargs)

# clean w/ config loader

class HiC(HumansDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__ == "__main__":
    def unfeat_extract(imgs):
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        imgs = imgs * std + mean
        imgs = imgs * 255
        return Image.fromarray(imgs.numpy().astype(np.uint8))
    
    def _un(img, mode=None):
        if mode != 'mask':
            img = (img + 1) * 255.0 / 2
        else:
            img = img * 255.0
        img = img.astype(np.uint8)
        return Image.fromarray(img)

    wandb.init(project="affordance")

    test_data = HiCRandBoxTest()
    all_imgs = []
    for idx in range(len(test_data)):      
        data = test_data[idx]
        all_imgs = [data['image'], data['masked_image'], data['refer_person'], data['mask']]
        wandb.log({"training_samples": [wandb.Image(x) for x in all_imgs]}, step=idx) 
        if idx == 200:
            break
