#!/usr/bin/env python3
# -- coding: utf-8 --

# import some common libraries
import numpy as np
import tqdm
import cv2
import time
from tqdm import tqdm

import os
import sys
import pickle
from pathlib import Path

import data_hic as data
import data_utils as utils

import lmdb

dataset = sys.argv[1]
path = Path(sys.argv[2])
split = int(sys.argv[3])

def encode(value):
    value_bytes = pickle.dumps(value)
    return value_bytes

def decode(value_bytes):
    value = pickle.loads(value_bytes)
    return value
    
def merge_lmdb(lmdbs, result_lmdb, merge_type):
    result_env = lmdb.open(result_lmdb, map_size=1024 ** 4)
    result_txn = result_env.begin(write=True)
    for idx, curr_lmdb in tqdm(enumerate(lmdbs)):
        curr_env = lmdb.open(curr_lmdb)
        curr_txn = curr_env.begin()
        curr_database = curr_txn.cursor()
        count = 0
        for (key, value) in tqdm(curr_database):
            if merge_type == 'k':
                key = key.decode()
                key = f'{idx}_{key}'
                key = key.encode()
                result_txn.put(key, value)
                # result_txn.delete(key)
            elif merge_type == 'kv':
                key = key.decode()
                key = f'{idx}_{key}'
                key = key.encode()
                value = decode(value)
                value = [f'{idx}_{x}' for x in value]
                value = encode(value)
                result_txn.put(key, value)
                # result_txn.delete(key)
            else:
                raise ValueError
            count += 1
            if (count%1000 == 0):
                result_txn.commit()
                count = 0
                result_txn = result_env.begin(write=True)
        if (count%1000 != 0):
            result_txn.commit()
            count=0
            result_txn=result_env.begin(write=True)
        curr_env.close()
    result_env.close()
  
all_data = [x for x in os.listdir(path) if x.startswith(dataset)]
all_data.sort()
if dataset in all_data:
    all_data.remove(dataset)
for data in all_data:
    if 'masks' in data:
        all_data.remove(data)
    if 'split' in data:
        all_data.remove(data)
if split == 0:
    all_data = all_data[:len(all_data) // 2]
else:
    all_data = all_data[len(all_data) // 2:]

for mode in ['frames_db', 'clipmask_db', 'masks_db', ]:
    lmdbs = []
    for curr_lmdb in all_data:
        lmdbs.append(str(path / curr_lmdb / mode))
    # TODO insert output path here
    output_path = Path('/home/output/path')
    result_lmdb = output_path / f'{dataset}_split{split}' / mode
    result_lmdb.mkdir(exist_ok=True, parents=True)
    result_lmdb = str(result_lmdb)
    if mode == 'clipmask_db':
        merge_lmdb(lmdbs, result_lmdb, 'kv')
    else:
        merge_lmdb(lmdbs, result_lmdb, 'k')
