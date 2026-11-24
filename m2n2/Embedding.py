from functools import partial
from typing import Callable

import torch
import torch.nn as nn


class Embedding:
    """
    A network and its training settings, read by fit: optimizer, the factory of its optimizer,
    called on the parameter groups of the network, and scheduler, the factory of its learning-rate
    scheduler, called on that optimizer (None = constant learning rate). They are set at construction,
    or later by MixedModelNN.set_optimizer and set_scheduler.

    network() must return one embedding per row of data, in the same order: aligning the two
    is the caller's responsibility. network(rows), with rows a boolean mask over the rows of
    data, must return the embeddings of the selected rows only: training feeds the labelled
    rows alone, so that batch statistics (e.g. BatchNorm) never see the others.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: Callable = partial(
            torch.optim.AdamW,
            lr = 1e-3,
            weight_decay = 1e-1
        ),
        scheduler: Callable | None = None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler

    def reset(self, seed: int = 42):
        """
        Draws new random weights, from `seed`, by reset_parameters() of every submodule that has one,
        the network included: BatchNorm running statistics are reset too. A parameter held by a module
        without reset_parameters() is left untouched. The global PyTorch generator is not affected.
        """
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for module in self.network.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def load(self, weights: str):
        """
        Loads the weights of the network in place, from a state_dict saved by torch.save, e.g. pretrained
        ones. The state_dict must match the network, buffers included.
        """
        self.network.load_state_dict(torch.load(weights, map_location="cpu"))
