from __future__ import annotations

__all__ = ["compute_pck"]

import torch.linalg as linalg

import os
import sys
import open_pose
import einops
import models
import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.io as io
import pathlib


class FileDataset:
    def __init__(self, gt_path, pred_path):
        self.gt_path = pathlib.Path(gt_path)
        gt_paths = os.listdir(self.gt_path)
        self.pred_path = pathlib.Path(pred_path)
        pred_paths = os.listdir(self.pred_path)
        pred_paths = ['000' + x[7:] for x in pred_paths]

        self.image_paths = set(gt_paths).intersection(set(pred_paths))
        self.image_paths = list(self.image_paths)
        self.image_paths.sort()

    def __getitem__(self, i):
        img_name = self.image_paths[i]

        gt_img = io.read_image(str(self.gt_path / img_name))
        gt_img = gt_img.to(torch.float)
        gt_img = 2.0 * gt_img / 255.0 - 1.0

        pred_img = io.read_image(str(self.pred_path / ('sample-' + img_name[-10:])))
        pred_img = pred_img.to(torch.float)
        pred_img = 2.0 * pred_img / 255.0 - 1.0

        return gt_img, pred_img

    def __len__(self):
        return len(self.image_paths)


@torch.no_grad()
def compute_pck_from_files(
    gt_path: str, pred_path: str, pose_model, batch_size: int = 32, \
    device: torch.device = torch.device("cpu"), threshold: float = 0.5
):
    dataset = FileDataset(gt_path, pred_path)

    progress_bar = tqdm.tqdm(total=len(dataset), desc="Predicting poses")

    correct_sum = torch.zeros(18, device=device)
    total_sum = torch.zeros(18, device=device)

    image_batch = []

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    for gt_images, pred_images in data_loader:
        gt_images = gt_images.to(device)
        pred_images = pred_images.to(device)

        keypoints_real = pose_model(gt_images)
        keypoints_fake = pose_model(pred_images)

        neck_position = keypoints_real[:, 0:1, :2]
        nose_position = keypoints_real[:, 1:2, :2]
        head_size = linalg.norm(neck_position - nose_position, dim=2)

        neck_visibility = keypoints_real[:, 0:1, 2]
        nose_visibility = keypoints_real[:, 1:2, 2]
        head_visibility = neck_visibility * nose_visibility

        threshold_scaled = threshold * head_size

        positions_real = keypoints_real[:, :, :2]
        positions_fake = keypoints_fake[:, :, :2]
        distance = linalg.norm(positions_real - positions_fake, dim=2)

        visibility_real = keypoints_real[:, :, 2]
        visibility_fake = keypoints_fake[:, :, 2]
        visibility = head_visibility * visibility_real * visibility_fake

        correct = (distance < threshold_scaled) * visibility
        correct = correct.sum(dim=0)
        correct_sum += correct

        total = head_visibility * visibility_real
        total = total.sum(dim=0)
        total_sum += total

        progress_bar.update(batch_size)

    progress_bar.close()

    pck = 100 * correct_sum / total_sum.clamp(min=1)
    pck_avg = pck.mean()

    pck = pck.tolist()
    pck_avg = float(pck_avg.item())

    return pck_avg

device = torch.device("cuda")
pose_model = open_pose.OpenPoseModelPChK().to(device)
pose_model = torch.nn.DataParallel(pose_model)

gt_dir = f'gt_dir_path/' # point to dir of GT images
pred_dir = f'pred_dir_path/' # point to dir of predicted samples
pck = compute_pck_from_files(
    str(gt_dir), str(pred_dir), pose_model, 1024, device, threshold = 0.5,
)
print(pck)
