from __future__ import annotations

import os
import time
import pathlib
import shutil
import sys
import re
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

import data_utils as utils
import data_hic as data
import open_pose


@dataclass
class FrameDataset(torch_data.Dataset):
    input_dir: pathlib.Path
    frame_keys: list[str]
    resolution: int = 256

    def __post_init__(self):
        frames_db_path = str(self.input_dir.joinpath("cropped_frames_db"))
        self.frames_db = data.ImageDatabase(frames_db_path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_key = self.frame_keys[index]
        frame = self.frames_db[frame_key]
        frame = data.to_tensor(frame)

        assert frame.size(1) == self.resolution
        assert frame.size(2) == self.resolution

        return {"key": frame_key, "frame": frame}

    def __len__(self) -> int:
        return len(self.frame_keys)


@dataclass
class FrameDataModule(pl.LightningDataModule):
    input_dir: pathlib.Path
    frame_keys: list[str]
    resolution: int = 256
    batch_size: int = 16
    num_workers: int = 3

    def __post_init__(self):
        super().__init__()

    def test_dataloader(self):
        # Broadcasts frame keys to all devices, since frame keys only present on
        # the main process.
        self._broadcast_frame_keys()

        dataset = FrameDataset(self.input_dir, self.frame_keys, self.resolution)
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
            desc="Detecting poses",
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )


@dataclass(eq=False)
class PoseDetector(pl.LightningModule):
    output_dir: pathlib.Path
    min_people: int = 1
    max_people: int = 1
    threshold_0: float = 0.1
    threshold_1: float = 0.05
    threshold_total_relaxed: float = 2.5
    threshold_total_strict: float = 10.0
    threshold_keypoint: float = 0.3
    min_keypoints: int = 8
    assert_ids: tuple[int, ...] = (1,)
    ignore_ids: tuple[int, ...] = (14, 15, 16, 17)

    def run(self, data_module: FrameDataModule, **kwargs):
        progress_callback = _ProgressCallback()
        trainer = pl.Trainer(callbacks=[progress_callback], **kwargs)
        trainer.test(self, datamodule=data_module)

    def __post_init__(self):
        super().__init__()
        self.pose_model = open_pose.OpenPoseModel(
            self.threshold_0, self.threshold_1
        )

        poses_db_path = str(self.output_dir.joinpath("poses_db"))
        self.poses_db = data.Database(poses_db_path, readonly=False)

        rejects_db_path = str(self.output_dir.joinpath("rejects_db"))
        self.rejects_db = data.Database(rejects_db_path, readonly=False)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        indices, poses = self._compute_poses(batch["frame"])

        poses_dict = {}
        for i, poses_i in zip(indices, poses):
            key = batch["key"][i]
            # self.poses_db[key] = poses_i.cpu()
            poses_dict[key] = poses_i.cpu()
        self.poses_db.add_batch(poses_dict)

        indices_all = range(len(batch["key"]))
        indices_rejects = list(set(indices_all) - set(indices))

        rejects_dict = {}
        for i in indices_rejects:
            key = batch["key"][i]
            # self.rejects_db.add(key)
            rejects_dict[key] = b""
        self.rejects_db.add_batch(rejects_dict)


    @torch.no_grad()
    def _compute_poses(
        self, images: torch.Tensor
    ) -> tuple[list[int], list[torch.Tensor]]:

        pose_sets = self.pose_model(images)

        indices = []
        keypoints_list = []

        for i, pose_set in enumerate(pose_sets):
            num_people = pose_set.num_people(self.threshold_total_relaxed)

            if num_people < self.min_people or num_people > self.max_people:
                continue

            keypoints = pose_set.get_poses(
                self.threshold_total_strict,
                self.threshold_keypoint,
                self.min_keypoints,
                self.assert_ids,
                self.ignore_ids,
            )
            if keypoints is None:
                continue

            num_people = keypoints.size(0)
            if num_people < self.min_people or num_people > self.max_people:
                continue

            indices.append(i)
            keypoints_list.append(keypoints)

        return indices, keypoints_list


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

    frames_db = data.ImageDatabase(str(input_dir.joinpath("cropped_frames_db")))

    def _open_db(db_name):
        db_path = output_dir.joinpath(db_name)
        if db_path.exists():
            return data.Database(str(db_path))

    poses_db = _open_db("poses_db")
    rejects_db = _open_db("rejects_db")

    frame_keys = []
    for frame_key in frames_db.keys(verbose=True):

        if poses_db and frame_key in poses_db:
            continue

        if rejects_db and frame_key in rejects_db:
            continue

        frame_keys.append(frame_key)

    num_poses = len(poses_db) if poses_db else 0
    num_rejects = len(rejects_db) if rejects_db else 0

    print(
        f"{num_poses} frames with pose detected; "
        f"{num_rejects} frames rejected; "
        f"{len(frame_keys)} frames remaining."
    )

    return frame_keys


@dataclass
class PoseClipMaker:
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    min_frames: int = 30
    split_on_frame_break: bool = False
    # quality: int = 90

    def __post_init__(self):
        frames_db_path = str(self.input_dir.joinpath("cropped_frames_db"))
        self.frames_db = data.ImageDatabase(frames_db_path)

        poses_db_path = str(self.output_dir.joinpath("poses_db"))
        self.poses_db = data.Database(poses_db_path)

        clip_frames_db_path = str(self.output_dir.joinpath("frames_db"))
        self.clip_frames_db = data.ImageDatabase(
            clip_frames_db_path, readonly=False, mode='png'
        )

        clip_poses_db_path = str(self.output_dir.joinpath("clip_poses_db"))
        self.clip_poses_db = data.Database(clip_poses_db_path, readonly=False)

        clips_db_path = str(self.output_dir.joinpath("clips_db"))
        self.clips_db = data.Database(clips_db_path, readonly=False)

    def run(self):
        frame_keys = self.poses_db.keys(verbose=True)
        frame_keys.sort()

        prev_clip_index = 0
        prev_frame_index = 0

        new_clips = []
        new_clip = []

        progress_bar = tqdm.tqdm(
            frame_keys, desc="Breaking valid frames into new clips"
        )

        for i, frame_key in enumerate(progress_bar):
            key_parts = frame_key.split("_")
            clip_index = int(key_parts[-3])
            frame_index = int(key_parts[-1])

            if (
                clip_index != prev_clip_index
                or (frame_index - prev_frame_index != 1 and self.split_on_frame_break)
            ):

                if len(new_clip) >= self.min_frames:
                    new_clips.append(new_clip)

                new_clip = []

            new_clip.append(frame_key)
            prev_clip_index = clip_index
            prev_frame_index = frame_index

            if i == len(frame_keys) - 1:
                if len(new_clip) >= self.min_frames:
                    new_clips.append(new_clip)

        progress_bar.close()

        # progress_bar = tqdm.tqdm(
        #     new_clips,
        #     desc="Saving clips with valid poses",
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

        frame_names = []

        clip_frames_dict = {}
        clip_poses_dict = {}
        for frame_index, frame_key in enumerate(clip):

            frame_name = f"{clip_name}_frame_{frame_index:010d}"
            frame_names.append(frame_name)

            frame = self.frames_db[frame_key]
            # self.clip_frames_db[frame_name] = frame
            clip_frames_dict[frame_name] = frame

            poses = self.poses_db[frame_key]
            # self.clip_poses_db[frame_name] = poses
            clip_poses_dict[frame_name] = poses

        self.clip_frames_db.add_batch(clip_frames_dict)
        self.clip_poses_db.add_batch(clip_poses_dict)
        self.clips_db[clip_name] = frame_names


# ==============================================================================
#
# ==============================================================================


@hydra.main(config_path="data_configs", config_name="detect_pose")
def filter_people(config: omegaconf.DictConfig):

    if config.input_dir is None or config.output_dir is None:
        raise AssertionError("Must specify input and output directories.")

    input_dir = pathlib.Path(config.input_dir)
    output_dir = pathlib.Path(config.output_dir)

    make_output_dir(output_dir)
    frame_keys = list_frame_keys(input_dir, output_dir)

    if not is_rank_zero() or len(frame_keys) > 0:
        data_module = FrameDataModule(input_dir, frame_keys, **config.data)
        person_filter = PoseDetector(output_dir, **config.pose_detector)
        person_filter.run(data_module, **config.backend)

    if is_rank_zero():
        time.sleep(60)
        clip_maker = PoseClipMaker(
            input_dir, output_dir, config.min_frames, config.split_on_frame_break
        )
        clip_maker.run()


if __name__ == "__main__":
    filter_people()
