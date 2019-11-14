# Standard import
import os

# Third-party import
from torch.utils.data import Dataset

# Local import
from invco import DATASET_DIR


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

class Recipe1M(Dataset):
    def __init__(self, root_dir):
        self.layer1, self.layer2 = load_layers()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
