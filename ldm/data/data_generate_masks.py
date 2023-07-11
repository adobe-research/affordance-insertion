from __future__ import annotations

import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import hydra
import omegaconf
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import simple_term_menu
import torch
import torch.distributed as dist
import torch.utils.data as torch_data
import tqdm
import numpy as np

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import data_utils as utils
import data_hic as data


@dataclass
class FrameDataset(torch_data.Dataset):
    input_dir: pathlib.Path
    frame_keys: list[str]
    input_format: str
    aug: T.augmentation.Augmentation
    resolution: int = 256

    def __post_init__(self):
        frames_db_path = str(self.input_dir.joinpath("frames_db"))
        self.frames_db = data.ImageDatabase(frames_db_path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_key = self.frame_keys[index]

        frame = self.frames_db[frame_key]
        frame = np.asarray(frame)

        if not self.input_format == "RGB":
            frame = frame[:, :, ::-1]
        else:
            frame = frame
        
        frame = self.aug.get_transform(frame).apply_image(frame)
        frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))

        # assert frame.size(1) == self.resolution
        # assert frame.size(2) == self.resolution

        return {"key": frame_key, "frame": frame}

    def __len__(self) -> int:
        return len(self.frame_keys)


@dataclass
class FrameDataModule(pl.LightningDataModule):
    input_dir: pathlib.Path
    frame_keys: list[str]
    input_format: str
    aug: T.augmentation.Augmentation
    resolution: int = 256
    batch_size: int = 16
    num_workers: int = 3

    def __post_init__(self):
        super().__init__()

    def test_dataloader(self):
        # Broadcasts frame keys to all devices, since frame keys only present on
        # the main process.
        self._broadcast_frame_keys()

        dataset = FrameDataset(self.input_dir, self.frame_keys, self.input_format, self.aug, self.resolution)
        data_loader = torch_data.DataLoader(
            dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return data_loader

    def _broadcast_frame_keys(self):
        object_list = [self.frame_keys]
        dist.broadcast_object_list(object_list, src=0)
        self.frame_keys = object_list[0]


# ==============================================================================
#
# ==============================================================================


class _ProgressCallback(callbacks.ProgressBar):
    def init_test_tqdm(self):
        return tqdm.tqdm(
            desc="Generating masks",
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )


@dataclass(eq=False)
class MaskGenerator(pl.LightningModule):
    output_dir: pathlib.Path
    cfg: detectron2.config.CfgNode
    threshold: float
    threshold_nms: float

    def run(self, data_module: FrameDataModule, **kwargs):
        progress_callback = _ProgressCallback()
        trainer = pl.Trainer(callbacks=[progress_callback], **kwargs)
        trainer.test(self, datamodule=data_module)

    def __post_init__(self):
        super().__init__()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.threshold_nms

        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        masks_db_path = str(self.output_dir.joinpath("masks_db"))
        self.masks_db = data.Database(masks_db_path, readonly=False)

        rejects_db_path = str(self.output_dir.joinpath("rejects_db"))
        self.rejects_db = data.Database(rejects_db_path, readonly=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        bs = batch['frame'].shape[0]
        all_frames = []
        for i in range(bs):
            all_frames.append({"image": batch['frame'][i], "height": 256, "width": 256})
        all_outputs = self.model(all_frames)

        masks_dict = {}
        for i in range(bs):
            outputs = all_outputs[i]
            pred_classes = outputs["instances"].pred_classes
            pred_boxes = outputs["instances"].pred_boxes
            pred_masks = outputs["instances"].pred_masks
            pred_scores = outputs["instances"].scores
            idx_ppl = (pred_classes == 0).nonzero()
            # if no ppl, skip
            if idx_ppl.shape[0] == 0:
                continue
            # select best person
            best_idx, best_score = None, float("-inf")
            for curr_idx in idx_ppl:
                curr_idx = curr_idx[0].item()
                curr_score = pred_scores[curr_idx] * pred_boxes[curr_idx].area()[0]
                if curr_score > best_score:
                    best_score = curr_score
                    best_idx = curr_idx
            idx_ppl = best_idx

            pred_masks = pred_masks[idx_ppl].cpu().numpy()
            pred_boxes = pred_boxes[idx_ppl]
            pred_centers = pred_boxes.get_centers().cpu().numpy()[0]
            pred_boxes = pred_boxes.tensor.cpu().numpy()[0]
            key = batch["key"][i]
            masks_dict[key] = {'mask': np.packbits(pred_masks), \
                                'box': pred_boxes, \
                                'center': pred_centers}
        self.masks_db.add_batch(masks_dict)

        indices_all = range(len(batch["key"]))
        indices_rejects = list(set(indices_all) - set(masks_dict.keys()))

        rejects_dict = {}
        for i in indices_rejects:
            key = batch["key"][i]
            # self.rejects_db.add(key)
            rejects_dict[key] = b""
        self.rejects_db.add_batch(rejects_dict)


# ==============================================================================
#
# ==============================================================================


def is_rank_zero():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank == 0


def make_output_dir(output_dir: pathlib.Path):
    if not is_rank_zero():
        return

    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")

        line = "Please select how to procede:"
        print(line)

        options = ("Resume", "Overwrite", "Exit")
        index = simple_term_menu.TerminalMenu(options).show()
        assert isinstance(index, int)

        # Appends the selected option to the end of the previous line.
        print(f"\033[F\033[{len(line)}C", options[index])

        if options[index].lower() == "exit":
            sys.exit(1)

        if options[index].lower() == "overwrite":
            print("Removing existing output directory...")
            shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True)


def list_frame_keys(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> list[str]:

    if not is_rank_zero():
        return []

    frames_db = data.ImageDatabase(str(input_dir.joinpath("frames_db")))

    def _open_db(db_name):
        db_path = output_dir.joinpath(db_name)
        if db_path.exists():
            return data.Database(str(db_path))

    masks_db = _open_db("masks_db")
    rejects_db = _open_db("rejects_db")

    frame_keys = []
    for frame_key in frames_db.keys(verbose=True):

        if masks_db and frame_key in masks_db:
            continue

        if rejects_db and frame_key in rejects_db:
            continue

        frame_keys.append(frame_key)

    num_masks = len(masks_db) if masks_db else 0
    num_rejects = len(rejects_db) if rejects_db else 0

    print(
        f"{num_masks} frames with mask detected; "
        f"{num_rejects} frames rejected; "
        f"{len(frame_keys)} frames remaining."
    )

    return frame_keys


@dataclass
class MaskClipMaker:
    input_dir: pathlib.Path
    output_dir: pathlib.Path

    def __post_init__(self):
        masks_db_path = str(self.output_dir.joinpath("masks_db"))
        masks_db = data.Database(masks_db_path)
        self.frame_keys = masks_db.keys(verbose=True)
        self.frame_keys.sort()

        clips_db_path = str(self.input_dir.joinpath("clips_db"))
        self.clips_db = data.Database(clips_db_path)

        clipmask_db_path = str(self.output_dir.joinpath("clipmask_db"))
        self.clipmask_db = data.Database(clipmask_db_path, readonly=False)

    def run(self):
        clips = self.clips_db.keys()

        with utils.ParallelProgressBar(n_jobs=-1) as parallel:
            parallel.tqdm(
                desc="Saving updated clip indices", unit=" pieces"
            )
            parallel(self._clip_worker, [clips[i::1000] for i in range(1000)])

    def _clip_worker(self, index: int, clips: List[str]):      
        result = {}
        for clip in clips:
            frames = self.clips_db[clip]
            new_frames = []
            for frame in frames:
                if frame in self.frame_keys:
                    new_frames.append(frame)
            result[clip] = new_frames
        self.clipmask_db.add_batch(result)


# ==============================================================================
#
# ==============================================================================


@hydra.main(config_path="data_configs", config_name="generate_masks")
def generate_mask(config: omegaconf.DictConfig):

    if config.input_dir is None or config.output_dir is None:
        raise AssertionError("Must specify input and output directories.")

    input_dir = pathlib.Path(config.input_dir)
    output_dir = pathlib.Path(config.output_dir)

    make_output_dir(output_dir)
    frame_keys = list_frame_keys(input_dir, output_dir)

    if not is_rank_zero() or len(frame_keys) > 0:
        # detectron setup
        detectron_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(detectron_model))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron_model)
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        input_format = cfg.INPUT.FORMAT
        assert input_format in ["RGB", "BGR"], input_format

        data_module = FrameDataModule(input_dir, frame_keys, input_format, aug, **config.data)
        mask_gen = MaskGenerator(output_dir, cfg, **config.mask_generator)
        mask_gen.run(data_module, **config.backend)

    if is_rank_zero():
        clip_maker = MaskClipMaker(input_dir, output_dir)
        clip_maker.run()


if __name__ == "__main__":
    generate_mask()
