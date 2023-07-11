from __future__ import annotations

__all__ = ["OpenPoseModel", "PoseSet", "OpenPoseModelPChK"]

import pathlib
from typing import Optional

import torch
import torch.jit as jit
import torch.nn as nn

from . import pose_affinity


class PoseSet:
    # subset: [N, 20]
    #   0-17: Keypooint index in candidate (-1 for none).
    #   18: Total score.
    #   19: Total parts
    # candidate: x, y, score, id

    def __init__(self, candidate, subset):
        self.candidate = torch.tensor(candidate)
        self.subset = torch.tensor(subset)

    def num_people(self, min_total_score: float = 2.5) -> int:
        count = 0
        for group in self.subset:
            total_score = group[18]
            if total_score >= min_total_score:
                count += 1
        return count

    def get_poses(
        self,
        min_total_score: float = 10.0,
        min_keypoint_score: float = 0.3,
        min_keypoints: int = 8,
        assert_ids: tuple[int, ...] = (1,),
        ignore_ids: tuple[int, ...] = (14, 15, 16, 17),
    ) -> Optional[torch.Tensor]:

        poses = []
        for group in self.subset:

            total_score = group[18]
            if total_score < min_total_score:
                continue

            indices = group[:18].long()
            visibility = indices != -1

            scores = self.candidate[indices, 2] * visibility
            visibility &= scores >= min_keypoint_score

            if not visibility[list(assert_ids)].all():
                continue

            num_keypoints = visibility.sum()
            num_keypoints -= visibility[list(ignore_ids)].sum()

            if num_keypoints < min_keypoints:
                continue

            x = self.candidate[indices, 0] * visibility
            y = self.candidate[indices, 1] * visibility
            keypoints = torch.stack((x, y, visibility), dim=1)
            poses.append(keypoints)

        if len(poses) > 0:
            poses = torch.stack(poses)
            return poses


class OpenPoseModel(nn.Module):
    FILENAME: str = "pretrained/open_pose.pt"

    def __init__(self, threshold_0: float = 0.1, threshold_1: float = 0.05):
        super().__init__()
        self.threshold_0 = threshold_0
        self.threshold_1 = threshold_1

        path = str(pathlib.Path(__file__).parent.joinpath(self.FILENAME))
        self.model = jit.load(path)

    def forward(self, image: torch.Tensor) -> list[PoseSet]:
        assert image.dim() == 4
        assert image.size(1) == 3
        assert image.size(2) % self.model.scale_factor == 0
        assert image.size(3) % self.model.scale_factor == 0

        affinity, heatmap, mask = self.model(image, self.threshold_0)

        pose_sets = []
        for i in range(image.size(0)):
            pose_set = self._single_image_pose_set(
                affinity[i], heatmap[i], mask[i], self.threshold_1
            )
            pose_sets.append(pose_set)

        return pose_sets

    @staticmethod
    def _single_image_pose_set(
        affinity: torch.Tensor,
        heatmap: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.05,
    ) -> PoseSet:

        peak_list = []
        peak_count = 0
        for part in range(18):

            idx_y, idx_x = mask[part].nonzero(as_tuple=True)
            scores = heatmap[part, idx_y, idx_x]

            prev_peak_count = peak_count
            peak_count += scores.numel()
            ids = list(range(prev_peak_count, peak_count))

            peaks = zip(idx_x.tolist(), idx_y.tolist(), scores.tolist(), ids)
            peak_list.append(list(peaks))

        affinity = affinity.permute(1, 2, 0)
        affinity = affinity.detach().cpu().numpy()

        candidate, subset = pose_affinity.pose_affinity(
            threshold, affinity.shape[0], affinity, peak_list
        )

        return PoseSet(candidate, subset)

    def keypoints(self, images: torch.Tensor) -> torch.Tensor:

        empty_keypoints = images.new_zeros(18, 3)
        keypoints_list = []

        pose_sets = self.forward(images)
        for pose_set in pose_sets:

            keypoints = pose_set.get_poses(
                min_total_score=0,
                min_keypoint_score=0,
                min_keypoints=0,
                assert_ids=(),
                ignore_ids=(),
            )
            if keypoints is None:
                keypoints_list.append(empty_keypoints)

            else:
                scores = []
                for group in pose_set.subset:
                    scores.append(group[18])

                i = torch.argmax(torch.tensor(scores)).item()
                keypoints = keypoints[i].to(images.device)
                keypoints_list.append(keypoints)

        keypoints = torch.stack(keypoints_list)
        return keypoints

class OpenPoseModelPChK(OpenPoseModel):
    def __init__(self, threshold_0: float = 0.1, threshold_1: float = 0.05):
        super().__init__(threshold_0, threshold_1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dim() == 4
        assert images.size(1) == 3
        assert images.size(2) % self.model.scale_factor == 0
        assert images.size(3) % self.model.scale_factor == 0

        affinity, heatmap, mask = self.model(images, self.threshold_0)

        pose_sets = []
        for i in range(images.size(0)):
            pose_set = self._single_image_pose_set(
                affinity[i], heatmap[i], mask[i], self.threshold_1
            )
            pose_sets.append(pose_set)

        empty_keypoints = images.new_zeros(18, 3)
        keypoints_list = []
        
        for pose_set in pose_sets:

            keypoints = pose_set.get_poses(
                min_total_score=0,
                min_keypoint_score=0,
                min_keypoints=0,
                assert_ids=(),
                ignore_ids=(),
            )
            if keypoints is None:
                keypoints_list.append(empty_keypoints)

            else:
                scores = []
                for group in pose_set.subset:
                    scores.append(group[18])

                i = torch.argmax(torch.tensor(scores)).item()
                keypoints = keypoints[i].to(images.device)
                keypoints_list.append(keypoints)

        keypoints = torch.stack(keypoints_list)
        return keypoints
