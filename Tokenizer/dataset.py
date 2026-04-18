import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from common import get_base_dir
DATASET_NAME = "JeanKaddour/minipile"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/data"
MAX_SHARD = 14
DATA_DIR,_ = get_base_dir()
os.makedirs(DATA_DIR, exist_ok=True)



def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(start=0, step=1):
    parquet_paths = list_parquet_files()
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
