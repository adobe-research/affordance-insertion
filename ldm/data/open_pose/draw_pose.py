__all__ = ["draw_pose"]

import math
from typing import List

import cv2
import data_hic as data
import einops
import numpy as np
import torch

from . import pose_model, pose_affinity

_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]  # type: ignore


def draw_pose(
    image: torch.Tensor, pose_sets: List[pose_model.PoseSet]
) -> torch.Tensor:

    assert image.dim() == 4
    assert image.size(0) == len(pose_sets)
    assert image.size(1) == 3

    image = data.to_uint8(image)
    image = image[:, [2, 1, 0]]  # RGB -> BGR
    image = einops.rearrange(image, "n c h w -> n h w c")

    for i, pose_set in enumerate(pose_sets):
        image[i] = _draw_pose_single_image(image[i], pose_set)

    image = einops.rearrange(image, "n h w c -> n c h w")
    image = image[:, [2, 1, 0]]  # BGR -> RGB
    image = data.to_float(image)

    return image


def _draw_pose_single_image(
    image: torch.Tensor, pose_set: pose_model.PoseSet
) -> torch.Tensor:

    canvas = image.detach().cpu().numpy().copy()

    size = min(canvas.shape[:2])
    thikness = max(1, size // 256)

    for i in range(18):
        for n in range(len(pose_set.subset)):

            index = int(pose_set.subset[n][i])
            if index == -1:
                continue

            x, y = pose_set.candidate[index][0:2]
            cv2.circle(
                canvas,
                (int(x), int(y)),
                2 * thikness,
                _COLORS[i],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    for i in range(17):
        for n in range(len(pose_set.subset)):
            index = pose_set.subset[n][np.array(pose_affinity._LIMB_SEQ[i]) - 1]
            if -1 in index:
                continue

            canvas_copy = canvas.copy()
            Y = pose_set.candidate[index.long(), 0]
            X = pose_set.candidate[index.long(), 1]

            mX = X.mean()
            mY = Y.mean()

            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)),
                (int(length / 2), thikness),
                int(angle),
                0,
                360,
                1,
            )

            cv2.fillConvexPoly(
                canvas_copy, polygon, _COLORS[i], lineType=cv2.LINE_AA
            )

            canvas = cv2.addWeighted(canvas, 0.4, canvas_copy, 0.6, 0)

    image = torch.tensor(canvas, device=image.device)
    return image
