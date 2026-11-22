"""
Shared setup of the functional tests: the meiosim Arabidopsis panel (DTF1, DTF2, DTF3, 10k SNPs), on CPU,
against the trivariate GBLUP reference of tests/reference/GBLUP.py. 20 % of the phenotyped accessions are
masked on every trait, and predicted by m2n2 alone.
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from meiosim import Arabidopsis
from meiosim.Population import MISSING

from m2n2 import MixedModelNN, Embedding, ToyNN
from pyreml import Residual

REFERENCE = json.loads((Path(__file__).parent / "reference" / "GBLUP.json").read_text())
TRAITS = REFERENCE["traits"]
G = np.array(REFERENCE["G"]) * REFERENCE["mean_diag_K"]  # genetic covariance per individual
R = np.array(REFERENCE["R"])

pop = Arabidopsis(n_SNPs=10_000, seed=42)
data = pop.phenotypes[["individual", *TRAITS]].merge(pop.metadata[["individual", "country"]], on="individual")
labelled = data[TRAITS].notna().any(axis=1).to_numpy()

# dosages mean-imputed over the whole panel, as in the reference
z = pop.genotypes.astype(np.float32)
z[z == MISSING] = np.nan
z = torch.tensor(np.where(np.isnan(z), np.nanmean(z, axis=0), z)[labelled])
data = data[labelled].reset_index(drop=True)

# reference GEBV, one column per trait
gebv = np.column_stack([data["individual"].astype(str).map(REFERENCE["gebv"][trait]).to_numpy() for trait in TRAITS])

# 20 % of the phenotyped accessions masked on every trait, for pure prediction
test = np.zeros(len(data), dtype=bool)
test[np.random.default_rng(42).permutation(len(data))[: round(0.2 * len(data))]] = True

D = 64  # embedding dimension


class Embedder(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.nn = ToyNN(z.shape[1], n_embedding=D)
        self.register_buffer("z", z)

    def forward(self, rows=None):
        return self.nn.embed(self.z if rows is None else self.z[rows])


def fit(response, random, fixed="1", residual=None):
    torch.manual_seed(42)
    return MixedModelNN.from_dataframe(
        data.assign(**{trait: data[trait].where(~test) for trait in TRAITS}),
        response=response,
        fixed=fixed,
        embedding={
            "geno": Embedding(Embedder(z), optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-1)),
        },
        random=random,
        residual=Residual() if residual is None else residual,
        lr_beta=1e-2,
        lr_var=1e-2,
    ).fit(n_epoch=1000, batch_size=256, validation=0.2, patience=50, verbose=False)


def correlation(prediction, trait, rows):
    """Correlation between a prediction and the reference GEBV of a trait, on the rows `rows`."""
    return np.corrcoef(prediction[rows], gebv[rows, TRAITS.index(trait)])[0, 1]
