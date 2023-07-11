from __future__ import annotations

import atexit
import bisect
import math
import os
import pathlib
import pickle
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import hydra
import omegaconf
import PIL.Image
import simple_term_menu
import tqdm

# import utils
import data_utils as utils

@dataclass()
class FrameExtractor:
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    # quality: int = 90
    resolution: int = 256
    min_frames: int = 30
    # max_frames: int = 3000
    num_jobs: int = -1

    def run(self):
        clip_paths = list(self.relative_frame_paths.keys())
        with utils.ParallelProgressBar(n_jobs=self.num_jobs) as parallel:
            parallel.tqdm(
                desc="Extracting frames from frame folders", unit=" 100clips"
            )
            clip_paths_chunks = [clip_paths[i:i + 100] for i in range(0, len(clip_paths), 100)]
            frame_paths = parallel(self._clip_worker, clip_paths_chunks)

        frame_paths = [paths for paths in frame_paths if paths is not None]
        frame_paths = [paths for frame_paths_inner in frame_paths for paths in frame_paths_inner]
        frame_paths.sort()
        frame_paths = dict(frame_paths)

        print(
            f"Extracted valid frames from {len(frame_paths)} of "
            f"{len(self.relative_frame_paths)} video clips."
        )

        with open(self.frame_paths_file, "wb") as open_file:
            pickle.dump(frame_paths, open_file)

        print("Saved sorted dictionary of frame paths.")

    def __post_init__(self):
        self.frame_paths_file = os.path.join(self.output_dir, "frame_paths.pkl")
        self._make_output_dir()

        self.relative_frame_paths = self._get_relative_clip_paths()
        self.temp_dir = self._make_temp_dir()

    def _clip_worker(
        self, chunk_index: int, relative_clip_paths: list[str]
    ) -> Optional[tuple[str, list[str]]]:

        result = []
        for curr_idx, relative_clip_path in enumerate(relative_clip_paths):
            index = 100 * chunk_index + curr_idx

            clip_name = f"clip_{index:010d}"

            if len(self.relative_frame_paths) > 1000:
                subset_name = f"{(index // 1000):04d}"
                subset_dir = os.path.join(self.output_dir, subset_name)
                relative_frame_dir = os.path.join(subset_name, clip_name)
            else:
                subset_dir = self.output_dir
                relative_frame_dir = clip_name

            frame_dir = os.path.join(self.output_dir, relative_frame_dir)

            if os.path.exists(frame_dir):
                frame_names = os.listdir(frame_dir)
                frame_names.sort()

            else:
                temp_frame_dir = os.path.join(self.temp_dir, relative_frame_dir)
                os.makedirs(temp_frame_dir)

                try:
                    frame_names = self._extract_frames_single_clip(
                        relative_clip_path, temp_frame_dir
                    )
                except Exception as e:
                    print(e)
                    continue
                else:
                    os.makedirs(subset_dir, exist_ok=True)
                    os.rename(temp_frame_dir, frame_dir)

            result.append((relative_frame_dir, frame_names))
            # return relative_frame_dir, frame_names
        return result

    def _extract_frames_single_clip(
        self, relative_clip_path: str, frame_dir: str
    ) -> list[str]:

        frame_names = []
        size = None

        clip_path = self.input_dir.joinpath(relative_clip_path)
        names = self.relative_frame_paths[relative_clip_path]

        for index, name in enumerate(names):

            frame_path = str(clip_path.joinpath(name))
            frame = PIL.Image.open(frame_path)

            if index == 0:
                short_edge = min(frame.size)

                if short_edge < self.resolution:
                    raise AssertionError(
                        f"Frame folder resolution too low: {relative_clip_path}"
                    )

                if short_edge > self.resolution:
                    scale = self.resolution / short_edge
                    width = int(math.ceil(scale * frame.width))
                    height = int(math.ceil(scale * frame.height))
                    size = (width, height)

            if size is not None:
                frame = frame.resize(size, resample=PIL.Image.LANCZOS)

            frame_name = f"frame_{index:010d}.png"
            frame_names.append(frame_name)

            frame_path = os.path.join(frame_dir, frame_name)
            # frame.save(frame_path, quality=self.quality)
            frame.save(frame_path)

            # if index == self.max_frames - 1:
            #     break

        if len(frame_names) < self.min_frames:
            raise AssertionError(
                f"Frame folder has too few frames: {relative_clip_path}"
            )

        return frame_names

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            print(f"Output directory already exists: {self.output_dir}")

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
                shutil.rmtree(self.output_dir)

            elif os.path.exists(self.frame_paths_file):
                raise AssertionError(
                    "Cannot resume since frame paths file already exists:  "
                    f"{self.frame_paths_file}"
                )

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_relative_clip_paths(self) -> dict[str, str]:
        if not self.input_dir.is_dir():
            raise ValueError(f"Directory not found: {self.input_dir}")

        list_path = self.input_dir.joinpath("frame_paths.pkl")

        if list_path.exists():
            with open(list_path, "rb") as open_file:
                frame_paths = pickle.load(open_file)

        else:
            frame_paths = {}
            directories = [str(self.input_dir)]

            progress_bar = tqdm.tqdm(desc="Listing all frame paths")

            while directories:

                directory = directories.pop()
                for entry in os.scandir(directory):

                    if entry.is_dir(follow_symlinks=False):
                        directories.append(entry.path)

                    elif entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        path = pathlib.Path(entry).parent
                        relative_path = path.relative_to(self.input_dir)

                        if relative_path in frame_paths:
                            bisect.insort(
                                frame_paths[relative_path], entry.name
                            )
                        else:
                            frame_paths[relative_path] = [entry.name]

                        progress_bar.update(1)

            progress_bar.close()

            for folder_path, names in frame_paths.items():
                frame_paths[folder_path] = sorted(names)

            with open(list_path, "wb") as open_file:
                pickle.dump(frame_paths, open_file)

            print("Saved list of frame paths.")

        return frame_paths

    def _make_temp_dir(self) -> str:
        temp_dir = os.path.join(self.output_dir, "tmp")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        os.mkdir(temp_dir)

        def _remove_temp_dir():
            print("Removing temporary directry...")
            shutil.rmtree(temp_dir)

        atexit.register(_remove_temp_dir)

        return temp_dir


@hydra.main(config_path="data_configs", config_name="from_images")
def extract_frames(config: omegaconf.DictConfig):

    if config.input_dir is None or config.output_dir is None:
        raise AssertionError("Must specify input and output directories.")

    input_dir = pathlib.Path(config.input_dir)
    output_dir = pathlib.Path(config.output_dir)

    frame_extractor = FrameExtractor(input_dir, output_dir, **config.extractor)
    frame_extractor.run()


if __name__ == "__main__":
    extract_frames()
