#!/usr/bin/env python3
# Standard import
import os
import json

# Third-party import
import torch

# Local import
from invco import DATASET_DIR, ROOT_DIR
from utils.dataset import Recipe1M
from utils.dataset import split_dataset


if '__main__' == __name__:
    # dataset = Recipe1M(DATASET_DIR)
    # split_csv(F'{DATASET_DIR}/recipeid.csv')
    split_dataset(F'{DATASET_DIR}/recipeid.csv', F'{ROOT_DIR}/csv', lambda r: r[4-1] + '.csv')

