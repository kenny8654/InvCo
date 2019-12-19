import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

class MaskedCrossEntropyCriterion(_WeightedLoss):

    def __init__(self, ignore_index=[-100], reduce=None):
        super(MaskedCrossEntropyCriterion, self).__init__()
        self.padding_idx = ignore_index
        self.reduce = reduce

    def forward(self, outputs, targets):
        lprobs = nn.functional.log_softmax(outputs, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        for idx in self.padding_idx:
            # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
            targets[targets == idx] = 0

        nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
        if self.reduce:
            nll_loss = nll_loss.sum()

        return nll_loss.squeeze()