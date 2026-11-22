"""
Trivariate on DTF1, DTF2 and DTF3: a general RandomNN term, its covariance across the traits structured
as iid, diag, full, then fa; full residual covariance, as in the reference.
"""
import numpy as np
import pytest
import torch

from m2n2 import RandomNN
from pyreml import Residual
from common import TRAITS, G, R, D, data, test, fit, correlation


@pytest.fixture(scope="module", params=["iid", "diag", "full", "fa"])
def model(request):
    n_axes = 2 if request.param == "fa" else None
    return fit(
        TRAITS,
        [RandomNN(embedding="geno", left_hand=request.param, n_axes=n_axes)],
        residual=Residual(left_hand="full"),
    )


def test_variances(model):
    unit = model.varcomp()["unit"]  # one row per trait for the term, then for the residual
    m = len(TRAITS)
    for r in range(m):
        assert 0.3 <= unit.iloc[r] / G[r, r] <= 3
        assert 0.3 <= unit.iloc[m + r] / R[r, r] <= 3


def test_genetic_correlations(model):
    if model.random[0].left_hand in ("iid", "diag"):
        pytest.skip("no covariance across the traits")
    S = np.asarray(model.random[0].variance["sigma"])
    sd, sd_ref = np.sqrt(np.diag(S)), np.sqrt(np.diag(G))
    assert np.abs(S / np.outer(sd, sd) - G / np.outer(sd_ref, sd_ref)).max() < 0.3


def test_convergence_network(model):
    assert 10 <= np.argmin(model.score) <= 90


def test_convergence_REML(model):
    assert model.convergence


@pytest.mark.parametrize("trait", TRAITS)
def test_prediction_train(model, trait):
    train = data[trait].notna().to_numpy() & ~test
    prediction = model.random[0].prediction[:, TRAITS.index(trait)].cpu().numpy()
    assert correlation(prediction, trait, train) > 0.9


@pytest.mark.parametrize("trait", TRAITS)
def test_prediction_test(model, trait):
    prediction = model.random[0].prediction[:, TRAITS.index(trait)].cpu().numpy()
    assert correlation(prediction, trait, test) > 0.8


def test_dimensions(model):
    m = len(TRAITS)
    assert model.random[0].prediction.shape == (len(data), m)
    assert model.random[0].uhat.shape == (m, 1, D)
    assert np.asarray(model.random[0].variance["sigma"]).shape == (m, m)
    assert len(model.varcomp()) == 2 * m


def test_finite(model):
    assert torch.isfinite(model.random[0].prediction).all()
    assert np.isfinite(model.score).all()
    varcomp = model.varcomp().to_numpy()
    assert np.isfinite(varcomp).all()
    assert (varcomp > 0).all()


def test_internal_state(model):
    assert not model.embedding["geno"].network.training
    with torch.no_grad():
        norms = model.random[0].embed().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert len(model.loss) == len(model.score)
