"""
Learning-rate schedulers on DTF1, a general RandomNN term with the country as a fixed effect, no validation.
A linear schedule over 120 epochs, warmup of 20, is stopped at epoch 70: the scheduled rates are halved.
"""
from functools import partial

import pytest
import torch

from m2n2 import MixedModelNN, Embedding, RandomNN
from common import data, z, Embedder

LR_NN, LR_BETA, LR_VAR = 1e-3, 1e-2, 1e-2
schedule = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda e: e / 20 if e < 20 else (120 - e) / 100)


@pytest.fixture(scope="module", params=["general", "network", "both"])
def model(request):
    torch.manual_seed(42)
    return MixedModelNN(
        data,
        response="DTF1",
        fixed="1 + country",
        embedding={
            "geno": Embedding(
                Embedder(z),
                optimizer=partial(torch.optim.AdamW, lr=LR_NN, weight_decay=1e-1),
                scheduler=schedule if request.param in ("network", "both") else None,
            ),
        },
        random=[RandomNN(embedding="geno")],
    ).fit(
        n_epoch=70,
        batch_size=256,
        validation=None,
        verbose=False,
        full_REML=False,
        lr_beta=LR_BETA,
        lr_var=LR_VAR,
        scheduler=schedule if request.param in ("general", "both") else None,
    ), request.param


def test_scheduler(model):
    model, case = model
    network, general = model.optimizers
    factor_nn = 0.5 if case in ("network", "both") else 1
    factor_general = 0.5 if case in ("general", "both") else 1
    assert [g["lr"] for g in network.param_groups] == pytest.approx([factor_nn * LR_NN] * 2)
    assert [g["lr"] for g in general.param_groups] == pytest.approx([factor_general * LR_BETA, factor_general * LR_VAR])
