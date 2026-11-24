from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class Embedding:
    """
    A network and the factory of its optimizer, called on the parameter groups of the network.

    network() must return one embedding per row of data, in the same order: aligning the two
    is the caller's responsibility. network(rows), with rows a boolean mask over the rows of
    data, must return the embeddings of the selected rows only: training feeds the labelled
    rows alone, so that batch statistics (e.g. BatchNorm) never see the others.
    """
    network: nn.Module
    optimizer: Callable = partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-1)
    scheduler: Callable | None = None
