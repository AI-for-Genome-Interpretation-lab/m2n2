import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...] = (256, 128),
        n_embedding: int = 64,
        dropout: float = 0.5,
        input_dropout: float = 0.1,
    ):
        super().__init__()

        sizes = (n_features, *hidden)
        self.net = nn.Sequential(
            nn.Dropout(input_dropout),
            *[
                module
                for n_in, n_out in zip(sizes, sizes[1:])
                for module in (
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            ],
            nn.Linear(sizes[-1], n_embedding),
        )
        self.head = nn.Sequential(nn.Linear(n_embedding, 1), nn.Flatten(0))

    def embed(self, z):
        return self.net(z)

    def forward(self, z):
        return self.head(self.embed(z))
    
    # Optional: a stand alone fit method, so that the NN can be trained
    # and benchmarked alongside a M2N2 model that includes it.
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epoch: int = 1000,
        batch_size: int | None = 64,
        validation: float | None = None,
        patience: int | None = 50,
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-1),
        verbose: bool = True,
    ):
        """
        batch_size: rows per minibatch, None = full batch; the incomplete last batch is dropped.
        validation: fraction of the rows held out (None or 0 = none) to score each epoch by the MSE;
        the best state is restored at the end, and the training stops after `patience` epochs
        without improvement (never if patience is None).
        """
        optimizer = optimizer([
            {"params": [p for p in self.parameters() if p.ndim >= 2]},
            {"params": [p for p in self.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ])

        perm = torch.randperm(len(y), device=y.device)
        n_valid = 0 if validation is None else round(validation * len(y))
        valid, train = perm[:n_valid], perm[n_valid:]
        size = len(train) if batch_size is None else min(batch_size, len(train))

        with torch.no_grad():
            self.head[0].bias.fill_(y[train].mean())

        self.loss, self.score = [], []
        best_mse, best_epoch = float("inf"), 0
        for epoch in range(n_epoch):

            self.train()
            losses = []
            for b in train[torch.randperm(len(train), device=y.device)].split(size):
                if len(b) < size:
                    continue
                optimizer.zero_grad()
                loss = F.mse_loss(self(x[b]), y[b])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.loss.append(float(np.mean(losses)))
            message = f"epoch {epoch + 1}/{n_epoch}  train MSE {self.loss[-1]:.4f}"

            if n_valid:
                self.eval()
                with torch.no_grad():
                    self.score.append(F.mse_loss(self(x[valid]), y[valid]).item())
                message += f"  validation MSE {self.score[-1]:.4f}"

                if self.score[-1] < best_mse:
                    best_mse, best_epoch = self.score[-1], epoch
                    best = copy.deepcopy(self.state_dict())

            if verbose:
                print(message)
            if n_valid and patience is not None and epoch - best_epoch >= patience:
                break

        if n_valid:
            self.load_state_dict(best)
        self.eval()
        return self

