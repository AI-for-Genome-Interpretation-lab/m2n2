"""
Univariate on DTF1: a general RandomNN term, alone or with a country-specific one on the same network.
"""
import numpy as np
import pytest
import torch

from m2n2 import RandomNN
from common import G, R, D, data, test, fit, correlation

train = data["DTF1"].notna().to_numpy() & ~test


@pytest.fixture(scope="module", params=[None, "country"], ids=["general", "specific"])
def model(request):
    random = [RandomNN(embedding="geno")]
    if request.param:
        random.append(RandomNN(embedding="geno", group=request.param))
    return fit("DTF1", random)


def genetic_value(model):
    return sum(term.prediction for term in model.random).cpu().numpy()


def test_variances(model):
    unit = model.varcomp()["unit"]  # RandomNN terms, then the residual
    for value, reference in [
        (unit.iloc[:-1].sum(), G[0, 0]),
        (unit.iloc[-1], R[0, 0]),
    ]:
        assert 0.3 <= value / reference <= 3


def test_convergence_network(model):
    assert 10 <= np.argmin(model.score) <= 90


def test_convergence_REML(model):
    assert model.convergence


def test_prediction_train(model):
    assert correlation(genetic_value(model), "DTF1", train) > 0.9


def test_prediction_test(model):
    assert correlation(genetic_value(model), "DTF1", test) > 0.8


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
