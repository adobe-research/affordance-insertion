from __future__ import annotations

import math
import os
import pathlib
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import einops
import hydra
import numpy as np
import omegaconf
import PIL.Image
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import simple_term_menu
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as torch_data
import tqdm

import data_utils as utils
import data_hic as data


@dataclass
class FrameDataset(torch_data.Dataset):
    input_dir: pathlib.Path
    frame_paths_list: list[str]
    resolution: int = 256

    def __getitem__(self, index: int) -> dict[str, Any]:
        relative_frame_path = self.frame_paths_list[index]
        frame_path = os.path.join(self.input_dir, relative_frame_path)

        frame = PIL.Image.open(frame_path)

        scale_x = self.resolution / frame.width
        scale_y = self.resolution / frame.height
        scale = min(scale_x, scale_y)

        width = int(math.ceil(scale * frame.width))
        height = int(math.ceil(scale * frame.height))

        frame = frame.resize((width, height), resample=PIL.Image.LANCZOS)
        frame = torch.from_numpy(np.array(frame))
        frame = einops.rearrange(frame, "h w c -> c h w")
        frame = data.to_float(frame)

        pad_x0 = (self.resolution - width) // 2
        pad_x1 = self.resolution - width - pad_x0
        pad_y0 = (self.resolution - height) // 2
        pad_y1 = self.resolution - height - pad_y0

        padding = (pad_x0, pad_x1, pad_y0, pad_y1)
        frame = F.pad(frame, padding)

        scale = max(width / height, height / width)

        return {
            "path": relative_frame_path,
            "frame": frame,
            "scale": scale,
            "pad_x0": pad_x0,
            "pad_y0": pad_y0,
        }

    def __len__(self) -> int:
        return len(self.frame_paths_list)


@dataclass
class FrameDataModule(pl.LightningDataModule):
    input_dir: pathlib.Path
    frame_paths: list[str]
    resolution: int = 256
    batch_size: int = 16
    num_workers: int = 3

    def __post_init__(self):
        super().__init__()

    def test_dataloader(self):
        # Broadcasts frame paths to all devices, since frame paths only present
        # on the main process.
        self._broadcast_frame_paths()

        dataset = FrameDataset(
            self.input_dir, self.frame_paths, self.resolution
        )
        data_loader = torch_data.DataLoader(
            dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return data_loader

    def _broadcast_frame_paths(self):
        object_list = [self.frame_paths]
        dist.broadcast_object_list(object_list, src=0)
        self.frame_paths = object_list[0]


# ==============================================================================
#
# ==============================================================================


class _ProgressCallback(callbacks.ProgressBar):
    def init_test_tqdm(self):
        return tqdm.tqdm(
            desc="Filtering frames for people", leave=True, dynamic_ncols=True,
        )


@dataclass(eq=False)
class PersonFilter(pl.LightningModule):
    output_dir: pathlib.Path
    min_people: int = 1
    max_people: int = 1
    threshold_nms: float = 0.3
    threshold_relaxed: float = 0.95
    threshold_strict: float = 0.98
    box_min_relaxed: float = 0.01
    box_min_strict: float = 0.04
    box_max_strict: float = 0.8

    def run(self, data_module: FrameDataModule, **kwargs):
        progress_callback = _ProgressCallback()
        trainer = pl.Trainer(callbacks=[progress_callback], **kwargs)
        trainer.test(self, datamodule=data_module)

    def __post_init__(self):
        super().__init__()
        self.person_detector = data.PersonDetector(
            self.threshold_relaxed, self.threshold_nms
        )

        boxes_db_path = str(self.output_dir.joinpath("boxes_db"))
        self.boxes_db = data.Database(boxes_db_path, readonly=False)

        rejects_db_path = str(self.output_dir.joinpath("rejects_db"))
        self.rejects_db = data.Database(rejects_db_path, readonly=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        indices, boxes = self._compute_boxes(
            batch["frame"], batch["scale"], batch["pad_x0"], batch["pad_y0"]
        )

        boxes_dict = {}
        for i, boxes_i in zip(indices, boxes):
            path = batch["path"][i]
            boxes_dict[path] = boxes_i.cpu()
            # self.boxes_db[path] = boxes_i.cpu()
        self.boxes_db.add_batch(boxes_dict)

        indices_all = range(len(batch["path"]))
        indices_rejects = list(set(indices_all) - set(indices.tolist()))

        rejects_dict = {}
        for i in indices_rejects:
            path = batch["path"][i]
            # self.rejects_db.add(path)
            rejects_dict[path] = b""
        self.rejects_db.add_batch(rejects_dict)

    @torch.no_grad()
    def _compute_boxes(
        self,
        images: torch.Tensor,
        scales: torch.Tensor,
        pads_x0: torch.Tensor,
        pads_y0: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        # The first filter is for ensuring there are not too many people present
        # in the image, and therefore uses relaxed thresholds.
        outputs = self.person_detector(images)

        area = images.size(2) * images.size(3)
        box_area_min = self.box_min_relaxed * area
        box_area_scales = scales ** 2

        outputs = self.person_detector.filter_outputs(
            outputs, box_area_min=box_area_min, box_area_scales=box_area_scales
        )
        indices, outputs = self.person_detector.filter_indices(
            outputs, self.min_people, self.max_people
        )

        if len(indices) == 0:
            return torch.empty(0), []

        # The second filter is for ensuring there are enough people present in
        # the image, and therefore uses strict thresholds.
        box_area_min = self.box_min_strict * area
        box_area_max = self.box_max_strict * area

        outputs = self.person_detector.filter_outputs(
            outputs,
            threshold=self.threshold_strict,
            box_area_min=box_area_min,
            box_area_max=box_area_max,
            box_area_scales=box_area_scales[indices],
        )
        indices, outputs = self.person_detector.filter_indices(
            outputs, self.min_people, self.max_people, indices
        )

        if len(indices) == 0:
            return torch.empty(0), []

        boxes = self.person_detector.extract_boxes(outputs)
        for i, scale in enumerate(scales[indices]):
            boxes[i][:, [0, 2]] -= pads_x0[i]
            boxes[i][:, [1, 3]] -= pads_y0[i]
            boxes[i] *= scale

        return indices, boxes


@dataclass
class PersonClipMaker:
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    min_frames: int = 30
    split_on_frame_break: bool = True
    # quality: int = 90

    def __post_init__(self):
        boxes_db_path = str(self.output_dir.joinpath("boxes_db"))
        self.boxes_db = data.Database(boxes_db_path)

        rejects_db_path = str(self.output_dir.joinpath("rejects_db"))
        self.rejects_db = data.Database(rejects_db_path)

        cropped_frames_db_path = str(
            self.output_dir.joinpath("cropped_frames_db")
        )
        self.cropped_frames_db = data.ImageDatabase(
            cropped_frames_db_path, readonly=False, mode='png'
        )

        cropped_boxes_db_path = str(
            self.output_dir.joinpath("cropped_boxes_db")
        )
        self.cropped_boxes_db = data.Database(
            cropped_boxes_db_path, readonly=False
        )

        cropped_clips_db_path = str(
            self.output_dir.joinpath("cropped_clips_db")
        )
        self.cropped_clips_db = data.Database(
            cropped_clips_db_path, readonly=False
        )        

    def run(self):
        frame_paths = self.boxes_db.keys()
        frame_paths.sort()

        reject_frame_paths = self.rejects_db.keys()
        
        all_frame_paths = frame_paths + reject_frame_paths
        all_frame_paths.sort()

        progress_bar = tqdm.tqdm(
            all_frame_paths, desc="Pre-computing spacing of all clips"
        )

        prev_clip_path = None
        spacing = {}
        for i, frame_path in enumerate(progress_bar):
            frame_path = pathlib.Path(frame_path)
            clip_path = str(frame_path.parent)
            if clip_path != prev_clip_path:
                frame_index = int(re.sub("[^\\d]", "", frame_path.name))
                next_frame_path = pathlib.Path(all_frame_paths[i + 1])
                next_frame_index = int(re.sub("[^\\d]", "", next_frame_path.name))
                spacing[clip_path] = next_frame_index - frame_index
                prev_clip_path = clip_path

        progress_bar.close()

        prev_clip_path = None
        prev_frame_index = 0

        new_clips = []
        new_clip = []

        progress_bar = tqdm.tqdm(
            frame_paths, desc="Breaking valid frames into new clips"
        )

        for i, frame_path in enumerate(progress_bar):
            frame_path = pathlib.Path(frame_path)

            clip_path = str(frame_path.parent)
            frame_index = int(re.sub("[^\\d]", "", frame_path.name))

            if (
                clip_path != prev_clip_path
                or (frame_index - prev_frame_index != spacing[clip_path] and self.split_on_frame_break)
            ):

                if len(new_clip) >= self.min_frames:
                    new_clips.append(new_clip)

                new_clip = []

            new_clip.append((str(frame_path)))
            prev_clip_path = clip_path
            prev_frame_index = frame_index

            if i == len(frame_paths) - 1:
                if len(new_clip) >= self.min_frames:
                    new_clips.append(new_clip)

        progress_bar.close()

        # FIXME: Filter out clips in cropped_clips_db.

        # progress_bar = tqdm.tqdm(
        #     new_clips,
        #     desc="Saving cropped clips with valid people",
        #     unit=" clips",
        #     smoothing=0.01,
        # )

        # for index, clip in enumerate(progress_bar):
        #     self._clip_worker(index, clip)
    
        # progress_bar.close()

        with utils.ParallelProgressBar(n_jobs=-1) as parallel:
            parallel.tqdm(
                desc="Saving cropped clips with valid people", unit=" clips"
            )
            parallel(self._clip_worker, new_clips)


    def _clip_worker(self, index: int, clip: str):
        clip_name = f"clip_{index:010d}"

        # FIXME: Why doesnt this work?

        # if clip_name in self.cropped_clips_db:
        #     return

        frame_names = []

        center_x = 0.0
        center_y = 0.0

        for frame_path in clip:
            boxes = self.boxes_db[frame_path]
            center_x += boxes[:, [0, 2]].mean().item()
            center_y += boxes[:, [1, 3]].mean().item()

        center_x = int(round(center_x / len(clip)))
        center_y = int(round(center_y / len(clip)))

        cropped_frames_dict = {}
        cropped_boxes_dict = {}
        for frame_index, relative_frame_path in enumerate(clip):
            frame_path = str(self.input_dir.joinpath(relative_frame_path))

            frame_name = f"{clip_name}_frame_{frame_index:010d}"
            frame_names.append(frame_name)

            frame = PIL.Image.open(frame_path)
            resolution = min(frame.size)

            if frame_index == 0:
                x0 = max(center_x - resolution // 2, 0)
                y0 = max(center_y - resolution // 2, 0)

                x0 = min(x0, frame.width - resolution)
                y0 = min(y0, frame.height - resolution)

                x1 = x0 + resolution
                y1 = y0 + resolution

            frame = frame.crop((x0, y0, x1, y1))  # type: ignore

            # self.cropped_frames_db[frame_name] = frame
            cropped_frames_dict[frame_name] = frame

            boxes = self.boxes_db[relative_frame_path]
            boxes[:, [0, 2]] -= x0  # type: ignore
            boxes[:, [1, 3]] -= y0  # type: ignore

            # self.cropped_boxes_db[frame_name] = boxes
            cropped_boxes_dict[frame_name] = boxes

        self.cropped_frames_db.add_batch(cropped_frames_dict)
        self.cropped_boxes_db.add_batch(cropped_boxes_dict)
        self.cropped_clips_db[clip_name] = frame_names


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


def list_frame_paths(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> list[str]:

    if not is_rank_zero():
        return []

    frame_paths_file = input_dir.joinpath("frame_paths.pkl")

    if not frame_paths_file.exists():
        raise AssertionError(
            f"Frame paths file does not exist: {frame_paths_file}"
        )

    with open(frame_paths_file, "rb") as open_file:
        frame_paths_dict = pickle.load(open_file)

    def _open_db(db_name):
        db_path = output_dir.joinpath(db_name)
        if db_path.exists():
            return data.Database(str(db_path))

    boxes_db = _open_db("boxes_db")
    rejects_db = _open_db("rejects_db")

    progress_bar = tqdm.tqdm(desc="Listing frame paths")

    frame_paths = []
    for frame_dir, frame_names in frame_paths_dict.items():

        for frame_name in frame_names:
            progress_bar.update()

            frame_path = str(pathlib.Path(frame_dir).joinpath(frame_name))

            if boxes_db and frame_path in boxes_db:
                continue

            if rejects_db and frame_path in rejects_db:
                continue

            frame_paths.append(frame_path)

    progress_bar.close()

    num_boxes = len(boxes_db) if boxes_db else 0
    num_rejects = len(rejects_db) if rejects_db else 0

    print(
        f"{num_boxes} frames with valid people detected; "
        f"{num_rejects} frames rejected; "
        f"{len(frame_paths)} frames remaining."
    )

    return frame_paths


# ==============================================================================
#
# ==============================================================================


@hydra.main(config_path="data_configs", config_name="filter_people")
def filter_people(config: omegaconf.DictConfig):

    if config.input_dir is None or config.output_dir is None:
        raise AssertionError("Must specify input and output directories.")

    input_dir = pathlib.Path(config.input_dir)
    output_dir = pathlib.Path(config.output_dir)

    make_output_dir(output_dir)
    frame_paths = list_frame_paths(input_dir, output_dir)
    if not is_rank_zero() or len(frame_paths) > 0:
        data_module = FrameDataModule(input_dir, frame_paths, **config.data)
        person_filter = PersonFilter(output_dir, **config.person_filter)
        person_filter.run(data_module, **config.backend)

    if is_rank_zero():
        clip_maker = PersonClipMaker(
            input_dir, output_dir, config.min_frames, config.split_on_frame_break
        )
        clip_maker.run()


if __name__ == "__main__":
    filter_people()
