from __future__ import annotations

__all__ = ["Database", "ImageDatabase"]

import io
import pickle
from typing import Any, Iterator

import lmdb
import PIL.Image
import tqdm


class Database:
    def __init__(
        self,
        path: str,
        readonly: bool = True,
        meminit: bool = False,
        map_size: int = 1024 ** 4,
        **kwargs,
    ):
        self.path = path
        self.readonly = readonly
        self.meminit = meminit
        self.map_size = map_size

        kwargs["create"] = kwargs.get("create", not self.readonly)
        self.kwargs = kwargs

        self._open_lmdb_env()

    @staticmethod
    def encode(value: Any) -> bytes:
        value_bytes = pickle.dumps(value)
        return value_bytes

    @staticmethod
    def decode(value_bytes: bytes) -> Any:
        value = pickle.loads(value_bytes)
        return value

    def add(self, key: str):
        self[key] = b""

    def add_batch(self, batch: dict):
        with self.lmdb_env.begin(write=True) as txn:
            for key, value in batch.items():
                key_bytes = key.encode()
                value_bytes = self.encode(value)
                txn.put(key_bytes, value_bytes, dupdata=False)

    def keys(self, verbose: bool = False):
        with self.lmdb_env.begin() as txn:
            with txn.cursor() as cursor:
                keys_iterator = cursor.iternext(values=False)

                if verbose:
                    progress_bar = tqdm.tqdm(
                        keys_iterator, "Listing all database keys", len(self)
                    )
                    keys = [key.decode() for key in progress_bar]
                else:
                    keys = [key.decode() for key in keys_iterator]  # type: ignore

        return keys

    def __contains__(self, key: str) -> bool:
        key_bytes = key.encode()

        with self.lmdb_env.begin() as txn:
            value_bytes = txn.get(key_bytes)

        contains = value_bytes is not None
        return contains

    def __getitem__(self, key: str) -> Any:
        key_bytes = key.encode()

        with self.lmdb_env.begin() as txn:
            value_bytes = txn.get(key_bytes)

        if value_bytes is None:
            return KeyError(f"Key not present in database: {key}")

        value = self.decode(value_bytes)
        return value

    def __setitem__(self, key: str, value: Any):
        key_bytes = key.encode()
        value_bytes = self.encode(value)

        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key_bytes, value_bytes, dupdata=False)

    def __delitem__(self, key: str):
        key_bytes = key.encode()

        with self.lmdb_env.begin(write=True) as txn:
            txn.delete(key_bytes)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for key in self.keys():
            value = self[key]
            yield key, value

    def __len__(self):
        return self.lmdb_env.stat()["entries"]

    def __del__(self):
        self.lmdb_env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lmdb_env"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._open_lmdb_env()

    def _open_lmdb_env(self):
        self.lmdb_env = lmdb.open(
            self.path,
            map_size=self.map_size,
            readonly=self.readonly,
            meminit=self.meminit,
            **self.kwargs,
        )


class ImageDatabase(Database):
    def __init__(
        self,
        path: str,
        quality: int = 90,
        readonly: bool = True,
        meminit: bool = False,
        mode: str = 'jpg',
        map_size: int = 1024 ** 4,
        **kwargs,
    ):
        self.quality = quality
        self.mode = mode
        super().__init__(path, readonly, meminit, map_size, **kwargs)

    def encode(self, image: PIL.Image.Image) -> bytes:
        buffer = io.BytesIO()
        if self.mode == 'jpg':
            image.save(buffer, format="JPEG", quality=self.quality)
        else:
            image.save(buffer, format="PNG")
        value_bytes = buffer.getvalue()
        return value_bytes

    @staticmethod
    def decode(value_bytes: bytes) -> PIL.Image.Image:
        buffer = io.BytesIO(value_bytes)
        image = PIL.Image.open(buffer)
        return image
