#!/usr/bin/env python3
# Standard import
import os
import json

# Third-party import
import torch

# Local import
from invco import DATASET_DIR
from utils.dataset import Recipe1M

if '__main__' == __name__:
    dataset = Recipe1M(DATASET_DIR)
