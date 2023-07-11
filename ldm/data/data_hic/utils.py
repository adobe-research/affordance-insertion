__all__ = [
    "read_and_resize_image",
    "crop_image_and_keypoints",
    "to_tensor",
    "to_pil_image",
    "to_float",
    "to_uint8",
    "make_grid",
]

import math
from typing import Tuple

import einops
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.utils as utils


# TODO: Move to IO?
def read_and_resize_image(
    path: str, resolution: int, letterbox: bool = False
) -> Tuple[torch.Tensor, float, float]:

    image = PIL.Image.open(path)

    scale_x = resolution / image.width
    scale_y = resolution / image.height

    if letterbox:
        scale = min(scale_x, scale_y)
    else:
        scale = max(scale_x, scale_y)

    width = int(math.ceil(scale * image.width))
    height = int(math.ceil(scale * image.height))

    scale_x = width / image.width
    scale_y = height / image.height

    image = image.resize((width, height), resample=PIL.Image.LANCZOS)
    image = torch.from_numpy(np.array(image))
    image = einops.rearrange(image, "h w c -> c h w")
    image = to_float(image)

    return image, scale_x, scale_y


def crop_image_and_keypoints(
    image: torch.Tensor, keypoints: torch.Tensor, resolution: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    if keypoints[:, 2].any():
        center_x = keypoints[:, 0].sum() / keypoints[:, 2].sum()
        center_y = keypoints[:, 1].sum() / keypoints[:, 2].sum()

        center_x = int(round(center_x.item()))
        center_y = int(round(center_y.item()))

    else:
        center_x = image.size(2) // 2
        center_y = image.size(1) // 2

    x0 = max(center_x - resolution // 2, 0)
    y0 = max(center_y - resolution // 2, 0)

    x0 = min(x0, image.size(2) - resolution)
    y0 = min(y0, image.size(1) - resolution)

    keypoints[:, 0] -= x0
    keypoints[:, 1] -= y0

    x1 = x0 + resolution
    y1 = y0 + resolution
    image = image[:, y0:y1, x0:x1]

    return image, keypoints


def to_tensor(pil_image: PIL.Image.Image) -> torch.Tensor:
    image = torch.from_numpy(np.array(pil_image))
    image = einops.rearrange(image, "h w c -> c h w")
    image = to_float(image)
    return image


def to_pil_image(image: torch.Tensor) -> PIL.Image.Image:
    image = to_uint8(image)
    image = einops.rearrange(image, "c h w -> h w c")
    image = image.cpu().numpy()
    pil_image = PIL.Image.fromarray(image)
    return pil_image


def to_float(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float)
    image = 2.0 * image / 255.0 - 1.0
    return image


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    image = 255.0 * (image + 1.0) / 2.0
    image = image.round().clamp(0.0, 255.0)
    image = image.to(torch.uint8)
    return image


def make_grid(image: torch.Tensor, **kwargs) -> torch.Tensor:
    assert image.dim() == 4

    if image.dtype == torch.half:
        image = image.float()

    nrow = kwargs.pop("nrow", None)

    if nrow is None:
        batch_size = image.size(0)
        nrow = 1

        for nrow in range(int(batch_size ** 0.5), 1, -1):
            if batch_size % nrow == 0:
                break

    return utils.make_grid(image, nrow=nrow, **kwargs)
