from __future__ import annotations

__all__ = ["HumansDataset", "HumansDatasetSubset"]

import os
import random
import pathlib
from dataclasses import dataclass, field
from typing import ClassVar, Union, Optional
import numpy as np
from einops import rearrange, repeat

from PIL import Image

import torch
from torch.utils.data import ConcatDataset, Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from transformers import CLIPFeatureExtractor
import bisect
from scipy import ndimage

from . import utils
from .database import Database, ImageDatabase
from .augment import AugmentPipe
from .comodgan_mask import RandomMaskRect


@dataclass
class HumansDataset(ConcatDataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 256
    num_frames: int = 1
    spacing: int = 4
    flip_p: float = 0.5
    deterministic: bool = False
    split: Optional[str] = "train"
    num_keypoints: ClassVar[int] = 18
    test_length: ClassVar[int] = 50000
    seed: ClassVar[int] = 10000
    # masks are ratios of bbox, mask, scribble, random bbox
    # modes are train, test, overfit, swap, halves
    # train - [25, 25, 25, 25], dil 10
    # test - [50, 50, 0, 0], dil 0
    # overfit - [100, 0, 0, 0], dil 0
    # swap - [50, 50, 0, 0], dil 10
    # halves - [50, 50, 0, 0], dil 0
    config: Dict[str] = field(default_factory=dict)

    def __post_init__(self):
        print(self.config)
        assert self.config is not None
        assert self.split in (None, "train", "test")

        path = pathlib.Path(self.path)
        datasets = []
        self.subsets = []

        for subset_path in path.iterdir():
            self.subsets.append(subset_path.name)

            dataset = HumansDatasetSubset(
                subset_path,
                self.resolution,
                self.num_frames,
                self.spacing,
                self.deterministic,
                self.flip_p,
                self.config,
            )
            datasets.append(dataset)

        self.subsets.sort()
        super().__init__(datasets)

        length = self.cumulative_sizes[-1]
        generator = torch.Generator().manual_seed(self.seed)
        self.indices = torch.randperm(length, generator=generator).tolist()

        if self.split == "test":
            self.indices = self.indices[: self.test_length]
        elif self.split == "train":
            self.indices = self.indices[self.test_length :]

        if self.config['mode'] == 'overfit':
            self.indices = self.indices[:1]

    def __getitem__(self, index: int):
        if self.config['mode'] == 'overfit':
            return super().__getitem__(self.indices[index // 1000000])
        else:
            return super().__getitem__(self.indices[index])

    def __len__(self) -> int:
        if self.config['mode'] == 'overfit':
            return len(self.indices) * 1000000
        else:
            return len(self.indices)


@dataclass
class HumansDatasetSubset(Dataset):
    path: Union[str, os.PathLike[str]]
    resolution: int = 128
    num_frames: int = 1
    spacing: int = 4
    deterministic: bool = False
    num_keypoints: ClassVar[int] = 18
    flip_p: float = 0.5
    config: Dict[str] = field(default_factory=dict)

    def __post_init__(self):
        # self.flip = transforms.RandomHorizontalFlip(p=self.flip_p)
        path = pathlib.Path(self.path)

        frames_db_path = str(path.joinpath("frames_db"))
        print(f'loading {frames_db_path}')
        self.frames_db = ImageDatabase(
            frames_db_path, readahead=False, lock=False
        )
    
        clips_db_path = str(path.joinpath("clipmask_db"))
        clips_db = Database(clips_db_path, lock=False)
        
        boxes_db_path = str(path.joinpath("masks_db"))
        self.boxes_db = Database(boxes_db_path, readahead=False, lock=False)

        self.keys = []
        self.min_length = self.spacing * (self.num_frames - 1) + 1

        clip_keys = clips_db.keys()
        for clip_key in clip_keys:
            frame_keys = clips_db[clip_key]
            if len(frame_keys) >= 2:
                self.keys.append(frame_keys)

        if self.deterministic:
            self.seed = int.from_bytes(path.name.encode(), byteorder="big")
            self.seed &= 0xFFFFFFFF
        else:
            self.seed = None

        if 'clip' in self.config:
            print(f"Using CLIP {self.config['clip']}")
            self.feat_extract = CLIPFeatureExtractor.from_pretrained(self.config['clip'])
        else:
            print(f"Using CLIP openai/clip-vit-large-patch14")
            self.feat_extract = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.first_aug_pipe = AugmentPipe(\
            xflip=0, rotate90=0, xint=0, xint_max=0.125,
            scale=0.0, rotate=0.0, aniso=0.0, xfrac=0, scale_std=0.2, rotate_max=0.2, aniso_std=0.2, xfrac_std=0.125,
            brightness=0.2, contrast=0.2, lumaflip=0, hue=0, saturation=0.2, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=0.5,
            imgfilter=0.1, imgfilter_bands=[1,1,1,1], imgfilter_std=0.2,
            noise=0.1, cutout=0.0, noise_std=0.05, cutout_size=0.5,
        )
        self.second_aug_pipe = AugmentPipe(\
            xflip=0, rotate90=0, xint=0, xint_max=0.125,
            scale=0.4, rotate=0.4, aniso=0.2, xfrac=0, scale_std=0.2, rotate_max=0.2, aniso_std=0.2, xfrac_std=0.125,
            brightness=0.0, contrast=0.0, lumaflip=0, hue=0, saturation=0.0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=0.5,
            imgfilter=0.0, imgfilter_bands=[1,1,1,1], imgfilter_std=0.2,
            noise=0.0, cutout=0.2, noise_std=0.05, cutout_size=0.5,
        )

    def _process_person(self, frame, mask, bbox, center):
        # mask_01 3D also 0-1 to extract person
        mask_01 = mask[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        frame = np.asarray(frame).astype(np.float32)

        # data augmentation for refer person
        aug_frame, aug_mask = torch.tensor(frame), torch.tensor(mask_01)
        aug_frame = rearrange(aug_frame, 'h w c -> 1 c h w')
        aug_frame = 2 * aug_frame / 255.0 - 1
        aug_mask = rearrange(aug_mask, 'h w c -> 1 c h w')
        if self.config['mode'] == 'train' and self.config['augment']:
            aug_frame = self.first_aug_pipe(aug_frame)
        aug_frame = aug_frame * aug_mask
        shift_x = 128 - center[0]
        shift_y = 128 - center[1]
        aug_frame = torchvision.transforms.functional.affine(aug_frame, 
                        translate=[int(shift_x), int(shift_y)],
                        angle=0, scale=1, shear=0, fill=0.0)
        if self.config['mode'] == 'train' and self.config['augment']:
            aug_frame = self.second_aug_pipe(aug_frame)
        aug_frame = (aug_frame + 1) * 255.0 / 2
        aug_frame = torch.clamp(aug_frame, min=0.0, max=255.0)
        aug_frame = rearrange(aug_frame, '1 c h w -> h w c')
        person_rgb = aug_frame.numpy()

        return person_rgb

    def _process_frame(self, frame, mask, bbox, center):
        # mask_01 3D also 0-1 to extract person
        mask_01 = mask[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        frame = np.asarray(frame).astype(np.float32)

        mask_type = random.choices(\
            ['bbox', 'pmask', 'scribble', 'smallbox', 'largebox'], \
            weights=self.config['masks'], k=1)[0]

        if mask_type == 'bbox':
            mask_rgb = mask * 0
            if self.config['dilation_mode'] == 'random':
                dilation = random.randint(0, self.config['dilation'])
            elif self.config['dilation_mode'] == 'fixed':
                dilation = self.config['dilation']
            else:
                raise ValueError
            y_min = max(int(bbox[1] - dilation // 2), 0)
            y_max = max(int(bbox[3] + (dilation + 1) // 2), 0)
            x_min = max(int(bbox[0] - dilation // 2), 0)
            x_max = max(int(bbox[2] + (dilation + 1) // 2), 0)
            if self.config['mode'] == 'halves':
                if random.random() < 0.5:
                    y_min = (y_min + y_max) // 2
                else:
                    y_max = (y_min + y_max) // 2
            mask_rgb[y_min:y_max, x_min:x_max] = 1
        elif mask_type == 'smallbox':
            y_min, y_max, x_min, x_max = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])
            area = (x_max - x_min) * (y_max - y_min)
            percent = random.uniform(25, 100)
            area = area * percent / 100
            sq = np.sqrt(area)
            width_new = random.uniform(min(sq, (x_max - x_min)), (x_max - x_min))
            height_new = min(area / width_new, (y_max - y_min))
            x_min_new = random.randint(x_min, x_max - int(width_new))
            y_min_new = random.randint(y_min, y_max - int(height_new))
            mask_rgb = mask * 0
            mask_rgb[y_min_new : y_min_new + int(height_new), \
                x_min_new : x_min_new + int(width_new)] = 1
        elif mask_type == 'pmask':
            mask_rgb = mask
            struct = ndimage.generate_binary_structure(2, 1)
            if self.config['dilation_mode'] == 'random':
                dilation = random.randint(0, self.config['dilation'])
            elif self.config['dilation_mode'] == 'fixed':
                dilation = self.config['dilation']
            else:
                raise ValueError
            mask_rgb = ndimage.binary_dilation(mask_rgb, structure=struct, iterations=dilation)
            mask_rgb = mask_rgb.astype(np.uint8)
        elif mask_type == 'scribble':
            mask_rgb = mask * 0
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            size = max(y_max - y_min, x_max - x_min)
            rand_submask = RandomMaskRect(y_max - y_min, x_max - x_min, hole_range=[0.2, 0.9])
            mask_rgb[y_min:y_max, x_min:x_max] = 1 - rand_submask
        elif mask_type == 'largebox':
            mask_rgb = mask * 0
            y_min = int(bbox[1])
            y_max = int(bbox[3])
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            size = max(y_max - y_min, x_max - x_min) // 2

            ratio = random.randint(5, 40)
            y_min_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            y_max_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            x_min_len = size * (100 + ratio) / 100.0
            ratio = random.randint(5, 40)
            x_max_len = size * (100 + ratio) / 100.0

            y_min = max(0, int(center[1] - y_min_len))
            y_max = max(0, int(center[1] + y_max_len))
            x_min = max(0, int(center[0] - x_min_len))
            x_max = max(0, int(center[0] + x_max_len))
            mask_rgb[y_min:y_max, x_min:x_max] = 1
            
        mask_01 = mask_rgb[:, :, np.newaxis].astype(np.float32)
        mask_01 = np.repeat(mask_01, 3, axis=2)
        masked_frame = frame * (1 - mask_01) + np.full_like(frame, 127.5) * (mask_01)
        mask_rgb = mask_rgb * 255

        return frame, masked_frame, mask_rgb

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_keys = self.keys[index]

        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)
        else:
            self.generator = None

        index_start = torch.randint(
            0, len(frame_keys), (), generator=self.generator
        ).item()
        if self.config['mode'] == 'test' or self.config['mode'] == 'swap':
            index_start = 0

        frame_key = frame_keys[index_start]
        frame = self.frames_db[frame_key]
        assert frame.height == frame.width
    
        refer_frame_keys = frame_keys.copy()
        refer_frame_keys.remove(frame_key)
        index_refer = torch.randint(
            0, len(refer_frame_keys), (), generator=self.generator
        ).item()
        index_refer = frame_keys.index(refer_frame_keys[index_refer])

        # get main frame and refer frame info
        if self.config['mode'] == 'swap':
            index_refer = 0
        if self.config['mode'] == 'test':
            index_refer = len(frame_keys) - 1
        refer_img = self.frames_db[frame_keys[index_refer]]
        refer_dict = self.boxes_db[frame_keys[index_refer]]
        if self.config['mode'] == 'swap':    
            # refer_clip_idx = random.randint(0, len(self.keys) - 1)
            refer_clip_idx = len(self.keys) - 1 - index
            refer_img = self.frames_db[self.keys[refer_clip_idx][index_refer]]
            refer_dict = self.boxes_db[self.keys[refer_clip_idx][index_refer]]
        frame_dict = self.boxes_db[frame_key]
                
        refer_center = refer_dict['center']
        refer_mask = refer_dict['mask']
        refer_mask = np.unpackbits(refer_mask.reshape(256, -1), axis=-1)
        frame_center = frame_dict['center']
        frame_mask = frame_dict['mask']
        frame_mask = np.unpackbits(frame_mask.reshape(256, -1), axis=-1)
                      
        frame_bbox = frame_dict['box']
        refer_bbox = refer_dict['box']
        is_uncond = 0  

        if self.config['data_type'] == 'image':
            refer_person = self._process_person(frame, frame_mask, frame_bbox, frame_center)  
            frame, frame_masked, frame_mask = self._process_frame(frame, frame_mask, frame_bbox, frame_center)
        else:    
            frame, frame_masked, frame_mask = self._process_frame(frame, frame_mask, frame_bbox, frame_center)
            refer_person = self._process_person(refer_img, refer_mask, refer_bbox, refer_center)

        ### FIXME only if support for res needs to be added
        # if frame.height != self.resolution:
        #     pose[:, :2] *= self.resolution / frame.height

        #     size = (self.resolution, self.resolution)
        #     frame = frame.resize(size, resample=Image.LANCZOS)
        #     frame_masked = frame_masked.resize(size, resample=Image.LANCZOS)
        #     frame_mask = frame_mask.resize(size, resample=Image.LANCZOS)
        #     refer_person = refer_person.resize(size, resample=Image.LANCZOS)

        if random.random() < self.flip_p:            
            frame, frame_masked, frame_mask, refer_person = \
                np.fliplr(frame), np.fliplr(frame_masked), np.fliplr(frame_mask), \
                np.fliplr(refer_person)

        def _n(frame, mode=None):
            frame = np.array(frame).astype(np.float32)
            if mode != 'mask':
                frame = 2.0 * frame / 255.0 - 1.0
            else:
                frame = frame / 255.0
            return frame

        zero_person = np.full_like(np.array(refer_person), 127.5)
        zero_encoding = self.feat_extract(Image.fromarray(zero_person.astype(np.uint8)), return_tensors="pt")
        drop_type = random.random()
        if (drop_type <= 0.10 and self.config['mode'] == 'train'):
            refer_person = np.full_like(np.array(refer_person), 127.5)
            frame_masked = np.full_like(np.array(frame_masked), 127.5)
            frame_mask = np.full_like(np.array(frame_mask), 255)
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")
            is_uncond = 1
        elif (drop_type > 0.10 and drop_type <= 0.20 and self.config['mode'] == 'train') or self.config['zero_person']:
            refer_person = np.full_like(np.array(refer_person), 127.5)
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")
            is_uncond = 1
        else:
            batch_encoding = self.feat_extract(Image.fromarray(refer_person.astype(np.uint8)), return_tensors="pt")

        frame, frame_masked, frame_mask, refer_person, zero_person = \
                _n(frame), _n(frame_masked), _n(frame_mask, mode='mask'), \
                _n(refer_person), _n(zero_person)      

        example = {'image': frame, 'masked_image': frame_masked, 'mask': frame_mask, \
            'refer_person_clip': rearrange(batch_encoding['pixel_values'][0], 'c h w -> h w c'), \
            'refer_person': refer_person, \
            'zero_person_clip': rearrange(zero_encoding['pixel_values'][0], 'c h w -> h w c'), \
            'zero_person': zero_person, \
            'uncond_mask': is_uncond
            }

        return example

    def __len__(self):
        return len(self.keys)
