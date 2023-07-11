__all__ = ["PersonDetector"]

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models.detection as detection
import torchvision.ops as ops


class PersonDetector(nn.Module):
    def __init__(self, threshold: float = 0.95, threshold_nms: float = 0.3):
        super().__init__()

        self.model = detection.keypointrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=threshold,
            box_nms_thresh=threshold_nms,
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        images = (images + 1) / 2
        outputs = self.model(images)
        outputs = [self._filter_output(output) for output in outputs]
        return outputs

    @torch.no_grad()
    def filter_outputs(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        threshold: Optional[float] = None,
        box_area_min: Optional[float] = None,
        box_area_max: Optional[float] = None,
        box_area_scales: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:

        if box_area_scales is not None:
            assert box_area_scales.dim() == 1
            assert box_area_scales.size(0) == len(outputs)

        filtered_outputs = []
        for i, output in enumerate(outputs):

            if threshold is not None:
                indices = torch.where(output["scores"] >= threshold)[0]
                output = self._filter_output(output, indices)

            if box_area_min is not None:
                box_area = ops.box_area(output["boxes"])
                if box_area_scales is not None:
                    box_area *= box_area_scales[i]

                indices = torch.where(box_area >= box_area_min)[0]
                output = self._filter_output(output, indices)

            if box_area_max is not None:
                box_area = ops.box_area(output["boxes"])
                if box_area_scales is not None:
                    box_area *= box_area_scales[i]

                indices = torch.where(box_area <= box_area_max)[0]
                output = self._filter_output(output, indices)

            filtered_outputs.append(output)

        return filtered_outputs

    @staticmethod
    def filter_indices(
        outputs: List[Dict[str, torch.Tensor]],
        min_people: int = 1,
        max_people: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:

        if indices is not None:
            assert indices.dim() == 1
            assert indices.size(0) == len(outputs)

        filtered_indices = []
        filtered_outputs = []

        for i, output in enumerate(outputs):
            count = output["scores"].size(0)

            if count >= min_people and (
                max_people is None or count <= max_people
            ):
                filtered_outputs.append(output)

                if indices is None:
                    filtered_indices.append(i)
                else:
                    filtered_indices.append(indices[i].item())

        filtered_indices = torch.tensor(filtered_indices)
        return filtered_indices, filtered_outputs

    @staticmethod
    def extract_boxes(
        outputs: List[Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:

        boxes = [output["boxes"] for output in outputs]
        return boxes

    # ==========================================================================
    # Private methods.
    # ==========================================================================

    @staticmethod
    def _filter_output(
        output: Dict[str, torch.Tensor], indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        filtered_output = dict()
        for key in ("boxes", "scores"):

            value = output[key]
            if indices is not None:
                value = value[indices]

            filtered_output[key] = value

        return filtered_output
