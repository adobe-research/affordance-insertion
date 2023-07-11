
import atexit
import math
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import av
import hydra
import omegaconf
import PIL.Image
import simple_term_menu

import logging

logging.basicConfig()
av.logging.set_level(av.logging.PANIC)
logging.getLogger('libav').setLevel(logging.ERROR)

import data_utils as utils


@dataclass()
class FrameExtractor:
    input_dir: str
    output_dir: str
    # quality: int = 90
    resolution: int = 256
    min_frames: int = 30
    min_frame_rate: float = 23.9
    max_frame_rate: float = 30.0
    min_bits_per_pixel: float = 0.9
    num_jobs: int = -1

    def run(self):
        with utils.ParallelProgressBar(n_jobs=self.num_jobs) as parallel:
            parallel.tqdm(
                desc="Extracting frames from video clips", unit=" clips"
            )
            relative_clip_paths_chunks = [self.relative_clip_paths[i:i + 100] \
                            for i in range(0, len(self.relative_clip_paths), 100)]
            frame_paths = parallel(self._clip_worker, relative_clip_paths_chunks)

        frame_paths = [paths for paths in frame_paths if paths is not None]
        frame_paths = [paths for frame_paths_inner in frame_paths for paths in frame_paths_inner]
        frame_paths.sort()
        frame_paths = dict(frame_paths)

        print(
            f"Extracted valid frames from {len(frame_paths)} of "
            f"{len(self.relative_clip_paths)} video clips."
        )

        with open(self.frame_paths_file, "wb") as open_file:
            pickle.dump(frame_paths, open_file)

        print("Saved sorted dictionary of frame paths.")

    def __post_init__(self):
        self.frame_paths_file = os.path.join(self.output_dir, "frame_paths.pkl")
        self._make_output_dir()

        self.relative_clip_paths = self._get_relative_clip_paths()
        self.temp_dir = self._make_temp_dir()

    def _clip_worker(
        self, chunk_index: int, relative_clip_paths: str
    ) -> Optional[Tuple[str, List[str]]]:

        result = []
        for curr_idx, relative_clip_path in enumerate(relative_clip_paths):
            index = 100 *  chunk_index + curr_idx

            clip_name = f"clip_{index:010d}"

            if len(self.relative_clip_paths) > 1000:
                subset_name = f"{(index // 1000):010d}"
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
                clip_path = os.path.join(self.input_dir, relative_clip_path)
                temp_frame_dir = os.path.join(self.temp_dir, relative_frame_dir)
                os.makedirs(temp_frame_dir)

                try:
                    frame_names = self._extract_frames_single_clip(
                        clip_path, temp_frame_dir
                    )
                except:
                    continue
                else:
                    os.makedirs(subset_dir, exist_ok=True)
                    os.rename(temp_frame_dir, frame_dir)
            
            result.append((relative_frame_dir, frame_names))

        return result

    def _extract_frames_single_clip(
        self, clip_path: str, frame_dir: str
    ) -> List[str]:

        container = av.open(clip_path)
        clip = container.streams.video[0]

        spacing = 1
        framerate = clip.codec_context.framerate

        while framerate > self.max_frame_rate:
            spacing += 1
            framerate = clip.codec_context.framerate // spacing

        if framerate < self.min_frame_rate:
            # print(f"Video clip invalid frame rate: {clip_path}")
            raise AssertionError(f"Video clip invalid frame rate: {clip_path}")

        width = clip.codec_context.width
        height = clip.codec_context.height

        # Bits per pixel per second.

        if clip.codec_context.bit_rate is not None:
            bits_per_pixel = clip.codec_context.bit_rate / (width * height)
        else:
            bits_per_pixel = 1

        if bits_per_pixel < self.min_bits_per_pixel:
            # print(f"Video clip bit rate too low: {clip_path}")
            raise AssertionError(f"Video clip bit rate too low: {clip_path}")

        short_edge = min(width, height)

        if short_edge < self.resolution:
            # print(f"Video clip resolution too low: {clip_path}")
            raise AssertionError(f"Video clip resolution too low: {clip_path}")

        if short_edge > self.resolution:
            scale = self.resolution / short_edge
            width = int(math.ceil(scale * width))
            height = int(math.ceil(scale * height))
            size = (width, height)
        else:
            size = None

        frame_names = []

        for index, frame in enumerate(container.decode(clip)):
            if index % spacing != 0:
                continue

            frame = frame.to_image()

            if size is not None:
                frame = frame.resize(size, resample=PIL.Image.LANCZOS)

            frame_name = f"frame_{index:010d}.png"
            frame_names.append(frame_name)

            frame_path = os.path.join(frame_dir, frame_name)
            frame.save(frame_path)
            # frame.save(frame_path, quality=self.quality)

            # if index == self.max_frames - 1:
            #     break

        if len(frame_names) < self.min_frames:
            # print(f"Video clip too short: {clip_path}")
            raise AssertionError(f"Video clip too short: {clip_path}")

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

    def _get_relative_clip_paths(self):
        clip_paths_file = os.path.join(self.input_dir, "clip_paths.pkl")

        if os.path.exists(clip_paths_file):
            with open(clip_paths_file, "rb") as open_file:
                relative_clip_paths = pickle.load(open_file)

            print("Loaded source video clip paths.")

        else:
            extensions = (".avi", ".mkv", ".mov", ".mp4", ".wmv", ".flv", ".VOB", ".mpg")
            relative_clip_paths = utils.list_file_paths(
                self.input_dir, extensions
            )

            with open(clip_paths_file, "wb") as open_file:
                pickle.dump(relative_clip_paths, open_file)

            print("Saved source video clip paths.")

        return relative_clip_paths

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


@hydra.main(config_path="data_configs", config_name="from_videos")
def extract_frames(config: omegaconf.DictConfig):

    if config.input_dir is None or config.output_dir is None:
        raise AssertionError("Must specify input and output directories.")

    frame_extractor = FrameExtractor(**config)
    frame_extractor.run()


if __name__ == "__main__":
    extract_frames()