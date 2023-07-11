__all__ = ["MaskGenerator"]

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models.detection as detection
import torchvision.ops as ops

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class MaskGenerator(nn.Module):
    def __init__(self, threshold: float = 0.95, threshold_nms: float = 0.3):
        super().__init__()

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = threshold_nms
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
            )
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

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
