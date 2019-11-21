# Standard import
import os
import csv
import json
import time
import functools
import subprocess
from collections import defaultdict

# Third-party import
from torch.utils.data import Dataset
from tqdm import tqdm

# Local import
from invco import DATASET_DIR

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

def scandir_recursive(base_dir):
    for root, subdirs, files in os.walk(base_dir):
        print(root, subdirs, files)

def load_layers():
    _dir = F'{DATASET_DIR}/recipe1M_layers'
    with open(F'{_dir}/layer1.json') as l1:
        layer1 = json.load(l1)
    with open(F'{_dir}/layer2.json') as l2:
        layer2 = json.load(l2)
    return layer1, layer2

def split_dataset(src, dst, func):
    n_rows = file_len(F'{DATASET_DIR}/recipeid.csv')
    with open(src) as f:
        reader = csv.reader(f)
        header = next(reader)
        writer = {}
        for row in tqdm(reader, total=n_rows):
            key = func(row)
            if key not in writer:
                writer[key] = csv.writer(open(F'{dst}/{key}', 'w', newline=''))
            writer[key].writerow(row)

class Recipe1M(Dataset):
    def __init__(self, root_dir):
        self.layer1, self.layer2 = load_layers()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
