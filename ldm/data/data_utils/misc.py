__all__ = [
    "assert_shape",
    "list_file_paths",
    "ask_remove_path",
    "ParallelProgressBar",
]


import os
import shutil
import sys
from typing import Any, Callable, List, Optional, Tuple

import joblib
import torch
import torch.autograd as autograd
import tqdm


def assert_shape(input: torch.Tensor, size: Tuple[Optional[int], ...]):

    if input.dim() != len(size):
        raise AssertionError(
            f"Wrong number of dimensions: got {input.dim()}, expected {len(size)}"
        )

    for index, (dim, dim_assert) in enumerate(zip(input.size(), size)):

        if dim_assert is None:
            continue

        if dim != dim_assert:
            raise AssertionError(
                f"Wrong size for dimension {index}: got {dim}, expected {dim_assert}"
            )


def list_file_paths(root_dir: str, extensions: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")

    walk_progress_bar = tqdm.tqdm(
        desc=f"Listing all files with extensions {{{', '.join(extensions)}}}"
    )

    paths = []
    directories = [root_dir]

    while directories:

        directory = directories.pop()
        for entry in os.scandir(directory):

            if entry.is_dir(follow_symlinks=False):
                directories.append(entry.path)

            elif entry.name.lower().endswith(extensions):
                path = os.path.relpath(entry.path, root_dir)
                paths.append(path)
                walk_progress_bar.update(1)

    walk_progress_bar.close()

    paths.sort()
    return paths


def ask_remove_path(path: str):
    if os.path.exists(path):
        value = input(
            f"Path already exists: {path}\n"
            "Would you like to overwrite it? [yes/no]: "
        )
        if value.lower() in ("y", "yes"):
            shutil.rmtree(path)
        else:
            sys.exit(1)


def profiled_function(fn):
    def decorator(*args, **kwargs):
        with autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator


class ParallelProgressBar(joblib.Parallel):
    def tqdm(self, **kwargs):
        self._tqdm_kwargs = kwargs

    def __call__(self, function: Callable[[int, Any], Any], inputs: List):
        tqdm_kwargs = getattr(self, "_tqdm_kwargs", dict())

        tqdm_kwargs["total"] = tqdm_kwargs.get("total", len(inputs))
        tqdm_kwargs["dynamic_ncols"] = tqdm_kwargs.get("dynamic_ncols", True)

        with tqdm.tqdm(**tqdm_kwargs) as self._progress_bar:

            return super().__call__(
                joblib.delayed(function)(index, input)
                for index, input in enumerate(inputs)
            )

    def print_progress(self):
        self._progress_bar.n = self.n_completed_tasks
        self._progress_bar.refresh()
