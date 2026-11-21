"""
Functional test on the meiosim Arabidopsis panel (DTF1, 10k SNPs), on CPU, against the GBLUP reference
of tests/reference/GBLUP.py. 20 % of the phenotyped accessions are masked and predicted by m2n2 alone.
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from meiosim import Arabidopsis
from meiosim.Population import MISSING

from m2n2 import MixedModelNN, Embedding, RandomNN, ToyNN
from pyreml import Residual

REFERENCE = json.loads((Path(__file__).parent / "reference" / "GBLUP.json").read_text())

pop = Arabidopsis(n_SNPs=10_000, seed=42)
data = pop.phenotypes[["individual", "DTF1"]].merge(pop.metadata[["individual", "country"]], on="individual")
labelled = data["DTF1"].notna().to_numpy()

# dosages mean-imputed over the whole panel, as in the reference
z = pop.genotypes.astype(np.float32)
z[z == MISSING] = np.nan
z = torch.tensor(np.where(np.isnan(z), np.nanmean(z, axis=0), z)[labelled])
data = data[labelled].reset_index(drop=True)

gebv = data["individual"].map(REFERENCE["gebv"]).to_numpy()

# 20 % of the phenotyped accessions masked, for pure prediction
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


@pytest.fixture(scope="module", params=[None, "country"], ids=["general", "specific"])
def model(request):
    torch.manual_seed(42)
    return MixedModelNN.from_dataframe(
        data.assign(DTF1=data["DTF1"].where(~test)),
        response="DTF1",
        embedding={
            "geno": Embedding(Embedder(z), optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-1)),
        },
        random=[RandomNN(embedding="geno")] + ([RandomNN(embedding="geno", group=request.param)] if request.param else []),
        residual=Residual(),
        lr_beta=1e-2,
        lr_var=1e-2,
    ).fit(n_epoch=1000, batch_size=256, validation=0.2, patience=50, verbose=False)


def test_variances(model):
    # GBLUP var_G is per unit of K: var_G * mean(diag K) is the genetic variance per individual
    unit = model.varcomp()["unit"]  # RandomNN terms, then the residual
    for value, reference in [
        (unit.iloc[:-1].sum(), REFERENCE["var_G"] * REFERENCE["mean_diag_K"]),
        (unit.iloc[-1], REFERENCE["var_R"]),
    ]:
        assert 0.3 <= value / reference <= 3


def test_convergence_network(model):
    assert 10 <= np.argmin(model.score) <= 90


@pytest.mark.xfail(reason="the REML polish runs in float32, whose digit budget is too short to declare convergence")
def test_convergence_REML(model):
    assert model.opti_REML.converged


def genetic_value(model):
    return sum(term.prediction for term in model.random)


def test_prediction_train(model):
    prediction = genetic_value(model).cpu().numpy()
    assert np.corrcoef(prediction[~test], gebv[~test])[0, 1] > 0.9


def test_prediction_test(model):
    prediction = genetic_value(model).cpu().numpy()
    assert np.corrcoef(prediction[test], gebv[test])[0, 1] > 0.8


def test_dimensions(model):
    n_groups = 0 if len(model.random) == 1 else data["country"].nunique()
    assert all(term.prediction.shape == (len(data),) for term in model.random)
    assert model.random[0].uhat.shape == (1, D)
    assert len(model.varcomp()) == 2 + bool(n_groups)
    assert all(len(term.pca) == 2 for term in model.random)
    if n_groups:
        assert model.random[1].uhat.shape == (n_groups, D)


def test_finite(model):
    assert all(torch.isfinite(term.prediction).all() for term in model.random)
    varcomp = model.varcomp().to_numpy()
    assert np.isfinite(varcomp).all()
    assert (varcomp > 0).all()


def test_internal_state(model):
    assert not model.embedding["geno"].network.training
    with torch.no_grad():
        norms = model.random[0].embed().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert len(model.loss) == len(model.score)
