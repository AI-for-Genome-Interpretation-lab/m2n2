"""
Univariate on DTF1: a general RandomNN term, with the country as a fixed effect, then as a pyreml Random.
"""
import numpy as np
import pytest
import torch

from m2n2 import RandomNN
from pyreml import Random
from common import G, R, D, data, test, fit, correlation

train = data["DTF1"].notna().to_numpy() & ~test


@pytest.fixture(scope="module", params=["fixed", "random"])
def model(request):
    if request.param == "fixed":
        return fit("DTF1", [RandomNN(embedding="geno")], fixed="1 + country")
    return fit("DTF1", [RandomNN(embedding="geno"), Random(unit="country")])


def test_country(model):
    if len(model.random) == 1:
        assert any(term.startswith("country") for term in model.estimates["term"])
    else:
        assert len(model.random[1].table) == data.loc[train, "country"].nunique()  # levels of the fitted rows


def test_variances(model):
    # the reference has no country effect: part of its genetic variance goes to the country here
    varcomp = model.varcomp()
    assert 0.1 <= varcomp.loc["geno", "unit"] / G[0, 0] <= 3
    assert 0.3 <= varcomp.loc["R", "unit"] / R[0, 0] <= 3


def test_convergence_network(model):
    assert 10 <= np.argmin(model.score) <= 90


def test_convergence_REML(model):
    assert model.convergence


def test_prediction_train(model):
    assert correlation(model.random[0].prediction.cpu().numpy(), "DTF1", train) > 0.6


def test_prediction_test(model):
    assert correlation(model.random[0].prediction.cpu().numpy(), "DTF1", test) > 0.5


def test_dimensions(model):
    assert model.random[0].prediction.shape == (len(data),)
    assert model.random[0].uhat.shape == (1, D)
    assert len(model.random[0].pca) == 2
    assert len(model.varcomp()) == 1 + len(model.random)


def test_finite(model):
    assert torch.isfinite(model.random[0].prediction).all()
    varcomp = model.varcomp().to_numpy()
    assert np.isfinite(varcomp).all()
    assert (varcomp > 0).all()


def test_internal_state(model):
    assert not model.embedding["geno"].network.training
    with torch.no_grad():
        norms = model.random[0].embed().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert len(model.loss) == len(model.score)
